from xbbg import blp
df = blp.bdp("SPX Index", "PX_LAST")
print(type(df))
print(type(df).__module__)
import pandas as pd
print("pandas version:", pd.__version__)
print("is real pandas DataFrame:", isinstance(df, pd.DataFrame))
