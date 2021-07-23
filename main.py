import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from businesslicense import business_lic, active_lic
import businesslicense

def summary_newcode():

    column_name = st.session_state['radio']
    if     column_name == businesslicense.str_lcode:
        search_code = st.session_state.select1.split('-')[0]    
    else:
        search_code = st.session_state.select2.split('-')[0]
        
    figure,bar,output = new.find_code(search_code,old, column_name)
    
    with st.session_state['map']:
        st.map(new.mapdata)

    with st.session_state['text']:
        st.write(output)
        st.pyplot(bar)
        st.write(new.outdf)
        
    with st.session_state['chart']:
        st.pyplot(figure)

def init():
    ''' run in notebook to refresh data from source. Takes a long time, not to be run inside
    streamlit
    ''' 
    db_client = MongoClient()
    db_chi=db_client.chicago
    lic_history=db_chi.license

    old=business_lic(lic_history)
    old.pull_hist_lic()
    new=active_lic()
    new.pull_curr_lic()
	        
st.title('Chicago Business Intelligence')
db_client = MongoClient()
db_chi=db_client.chicago
lic_history=db_chi.license

old=business_lic(lic_history)
#old.compact()
old.load()
new=active_lic()
new.load()
#new.generate()

with st.form("my_form"):
    st.write("Input Selection")
    choice_list=[str(i)+'-'+new.lcode_dict[str(i)]['text']+'-'+str(new.lcode_dict[str(i)]['count'])+' licenses'   for i in range(0,9999) if str(i) in new.lcode_dict]
    input_choice=st.selectbox('Select a valid license code',choice_list,key='select1')

    choice_list2=[str(i)+'-'+new.b_id_dict[str(i)]['text']+'-'+str(new.b_id_dict[str(i)]['count'])+' licenses'   for i in range(-1,2000) if str(i) in new.b_id_dict]
    input_choice2=st.selectbox('Select a valid business activity ID',choice_list2,key='select2')
 
    checkbox_val = st.radio("Select license code or Business activity ID", (businesslicense.str_lcode, 
                      businesslicense.str_b_id),key='radio')
    # Every form must have a submit button.    
    submitted = st.form_submit_button("Submit")    



text_holder=st.beta_container()
st.session_state['text']=text_holder
map_image=st.empty()
st.session_state['map']=map_image
chart=st.empty()
st.session_state['chart']=chart

summary_newcode()
