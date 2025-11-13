import streamlit as st
from styles import apply_styles
def show():
    apply_styles()
    st.title("👨‍💻 About Me")
    
    st.markdown("""
    ## Hi there! I'm **Niharika Sharma**, a AI & Machine Learning Enthusiast 🚀
    
    ### 🌟 Key Areas of Interest:
    - Artificial Intelligence (AI)
    - Machine Learning (ML)
    
    ### 📬 Let's Connect:
    - **LinkedIn**: [Niharika Sharma](https://www.linkedin.com/in/niharika-sharma16/)
    - **GitHub**: [niha16](https://github.com/niha16)
    - **Email**: [niharikash16@gmail.com](mailto:niharikash16@gmail.com)

    Feel free to reach out for collaborations, learning, or discussions on anything AI & ML-related!
    """)

    st.markdown("___")
    st.markdown("I'm always excited to connect with like-minded individuals and share knowledge!")
