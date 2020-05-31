import util

confidence = util.loadEnvOrEmpty("THRESHOLD").replace("\"","")
confidence = 0 if confidence == '' else float(confidence)

util.recoverCropImageFromPath(util.loadEnv("TO_RECOVER"), confidence)