while True:
    import time
    localtime = time.localtime()
    result = time.strftime("%I:%M:%S %p", localtime)
    print(result)
    print("You are not allowed to run this code")
    time.sleep(1)