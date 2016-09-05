#include <FFT_UGens.h>
#include <dlfcn.h>
#include <stdio.h>
#include <libgen.h>
#include <string.h>

// InterfaceTable contains pointers to functions in the host (server).
InterfaceTable *ft;

// declare struct to hold unit generator state
struct LoadIt : public Unit
{
};

// declare unit generator functions
static void LoadIt_next_a(LoadIt *unit, int inNumSamples);
static void LoadIt_next_k(LoadIt *unit, int inNumSamples);
static void LoadIt_Ctor(LoadIt* unit);


//////////////////////////////////////////////////////////////////

// Ctor is called to initialize the unit generator.
// It only executes once.

// A Ctor usually does 3 things.
// 1. set the calculation function.
// 2. initialize the unit generator state variables.
// 3. calculate one sample of output.
void LoadIt_Ctor(LoadIt* unit)
{
}


//////////////////////////////////////////////////////////////////

// The calculation function executes once per control period
// which is typically 64 samples.

// calculation function for an audio rate frequency argument
void LoadIt_next_a(LoadIt *unit, int inNumSamples)
{
}

//////////////////////////////////////////////////////////////////

// calculation function for a control rate frequency argument
void LoadIt_next_k(LoadIt *unit, int inNumSamples)
{
}

typedef void (*LoadPlugInFunc)(struct InterfaceTable *);



// the entry point is called by the host when the plug-in is loaded
PluginLoad(LoadIt)
{
    // InterfaceTable *inTable implicitly given as argument to the load function
    ft = inTable; // store pointer to InterfaceTable
    char filename[1000];
    DefineSimpleUnit(LoadIt);	

    // This will sneakily load your python lib but it is hardcoded so watch out
    void* handlepy = dlopen( OURPYTHONLIB, RTLD_LAZY | RTLD_GLOBAL);
    if (!handlepy) {
        fprintf(stderr,"dlopen of pylib did not work: %s\n", dlerror());
        //scprintf("*** Loadit ERROR: dlopen '%s' err '%s'\n", filename, dlerror());
        dlclose(handlepy);
        return;
    }


    // This is pretty terrible, basically we try to load the REAL library
    // from the same directory as this module
    // So we need to ask where this module is being loaded from
    // and then we load our module from that location
    // we use a clever naming scheme .os -> .so 
    // this is because we want to use the RTLD_LAZY | RTLD_GLOBAL flags
    // on dlopen
    Dl_info dl_info;
    dladdr((void *)LoadIt_next_k, &dl_info);
    fprintf(stderr, "module %s loaded\n", dl_info.dli_fname);
    size_t len = strnlen(dl_info.dli_fname,sizeof(filename));
    strncpy( filename, dl_info.dli_fname, sizeof(filename));
    filename[len - 2] = 'o'; 
    filename[len - 1] = 's'; 
    // Convert "/home/hindle1/.local/share/SuperCollider/Extensions/scpy.so"
    // to      "/home/hindle1/.local/share/SuperCollider/Extensions/scpy.os"
    fprintf(stderr,"Trying to load %s\n", filename);
    void* handle = dlopen(filename, RTLD_LAZY | RTLD_GLOBAL );
    if (!handle) {
        fprintf(stderr,"dlopen did not work: %s\n", dlerror());
        dlclose(handle);
        return;
    }
    
    void *ptr = dlsym(handle, "load");
    if (!ptr) {
        fprintf(stderr,"dlsym on load did not work\n");
        dlclose(handle);
        return;
    }
    LoadPlugInFunc loadFunc = (LoadPlugInFunc)ptr;
    (*loadFunc)(ft);
    
    // open_handles.push_back(handle);
}
