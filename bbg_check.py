# bbg_check.py -- isolate the xbbg/blpapi connection
print("1) importing blpapi...")
try:
    import blpapi
    print("   blpapi OK, version:", blpapi.__version__)
except Exception as e:
    print("   blpapi IMPORT FAILED:", repr(e))

print("2) importing xbbg...")
from xbbg import blp

print("3) raw blpapi session to localhost:8194...")
try:
    opts = blpapi.SessionOptions()
    opts.setServerHost("localhost"); opts.setServerPort(8194)
    s = blpapi.Session(opts)
    print("   session.start():", s.start())
    print("   openService(//blp/refdata):", s.openService("//blp/refdata"))
    s.stop()
except Exception as e:
    print("   raw session FAILED:", repr(e))

print("4) xbbg bdp SPX Index PX_LAST...")
df = blp.bdp("SPX Index", "PX_LAST")
print("   shape:", df.shape, "empty:", df.empty)
print(df)
