#include <FFT_UGens.h>
#include <dlfcn.h>
#include <stdio.h>



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

    DefineSimpleUnit(LoadIt);	
    printf("Trying to load scpy.os\n");
    void* handle = dlopen("/home/hindle1/.local/share/SuperCollider/Extensions/scpy.os", RTLD_LAZY | RTLD_GLOBAL );
    if (!handle) {
        printf("dlopen did not work: %s\n", dlerror());
        //scprintf("*** Loadit ERROR: dlopen '%s' err '%s'\n", filename, dlerror());
        dlclose(handle);
        return;
    }
    
    void *ptr = dlsym(handle, "load");
    if (!ptr) {
        printf("dlsym on load did not work\n");
        //scprintf("*** ERROR: dlsym load err '%s'\n", dlerror());
        dlclose(handle);
        return;
    }
    LoadPlugInFunc loadFunc = (LoadPlugInFunc)ptr;
    (*loadFunc)(ft);
    
    // open_handles.push_back(handle);
}
