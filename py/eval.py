import api

def setup(code, context={}):
    context = dict(api.__dict__, **context)
    exec(code + "\n_newContext = locals()", context, context)
    return context['_newContext']

def run(code, context={}):
    context = dict(api.__dict__, **context)
    exec("_output = " + code, context, context)
    return context['_output']
