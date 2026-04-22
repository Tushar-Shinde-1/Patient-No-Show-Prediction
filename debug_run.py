import sys, traceback
try:
    import model_pipeline
    model_pipeline.main()
except Exception as e:
    with open("err_debug.txt", "w") as f:
        traceback.print_exc(file=f)
