"""Session login gate. Fixed shared password is not a public-repository secret."""
import hashlib
import hmac

import streamlit as st


def require_access():
    if st.session_state.get("site_authenticated") is True:
        with st.sidebar:
            st.caption("股票100% · 已验证")
            if st.button("退出登录", key="logout"):
                # Clear this user's report, packet and encoded key, not other sessions.
                st.session_state.clear()
                st.rerun()
        return
    with st.container(border=True, key="login_panel"):
        st.subheader("访问验证")
        st.caption("请输入访问密码后进入股票分析看板。")
        with st.form("login_form", clear_on_submit=True):
            password = st.text_input("访问密码", type="password")
            submitted = st.form_submit_button("进入看板", type="primary")
        if submitted:
            actual = hashlib.sha256(password.encode("utf-8")).digest()
            expected = hashlib.sha256(b"xlx222").digest()
            if hmac.compare_digest(actual, expected):
                st.session_state["site_authenticated"] = True
                st.rerun()
            st.error("密码不正确，请重新输入。")
    st.stop()
