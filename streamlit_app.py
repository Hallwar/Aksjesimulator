import sys, streamlit as st, numpy, pandas, matplotlib
st.title("Sanity check")
st.write({
    "python": sys.version.split()[0],
    "streamlit": st.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "matplotlib": matplotlib.__version__,
})
st.success("Miljøet kjører ✅")
