"""
Home.py — entrypoint for the multipage app.

Streamlit turns every file in pages/ into a nav item automatically, ordered by
the numeric prefix. Adding a second tool later means dropping in
pages/2_Whatever.py and nothing else.

Only the entrypoint may call st.set_page_config, which is why the page files
do not.
"""

import streamlit as st

st.set_page_config(
    page_title="Tragic Algebra & IV15 Analyzer — Michael Burry Owners' Earnings Calculator",
    page_icon="🎯",
    layout="centered",
    # Collapsed by default: the nav is one line and should not greet anyone.
    initial_sidebar_state="collapsed",
)

st.title("🎯 Tragic Algebra Analyzer")
st.caption("Owners' earnings after the true cost of stock compensation, "
           "then the price ladder that follows")

st.markdown(
    """
Reported profit is not what reaches you. Shares handed to employees cost real money the
income statement never shows — either through dilution, or through buybacks that exist only
to offset employee grants.

Burry's study of the NASDAQ-100 over the ten years to 2025 put that cost at **$1.73 trillion**,
leaving shareholders about **83 cents** of every reported GAAP dollar. That figure is his,
for that period. This tool computes the same thing for any company, from whatever the filings
say today.

Michael Burry's Tragic Algebra and IV15 framework, run directly on audited SEC filings.
"""
)

st.page_link("pages/1_Tragic_Algebra_Analyzer.py",
             label="**Open the analyzer**", icon="🎯")

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "**Single stock**\n\n"
        "Enter a ticker for owners' earnings, the full IV ladder from IV8 to IV20, a stress "
        "test, and the shareholder-quality verdict."
    )
with c2:
    st.markdown(
        "**Watchlist**\n\n"
        "Screen up to 25 tickers ranked by ΔE, the share of reported profit that actually "
        "survives. No judgement needed — it is arithmetic on the filings."
    )

st.divider()
st.caption(
    "Research aid, not financial advice. Outputs depend on estimates you supply — change the "
    "growth rate and the answer changes a great deal. Method follows Michael Burry's published "
    "writing; this project is independent and is not affiliated with or endorsed by him or "
    "Scion Asset Management."
)
