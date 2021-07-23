from sodapy import Socrata
from datetime import datetime,date
import pandas as pd
import numpy as np
import pickle
import time
import pymongo 
from pymongo import MongoClient
import matplotlib.pyplot as plt
import streamlit as st
'''
Create pipeline for processing Chicago business licenses.
First process licenses in previous years and obtain historical data
Then process current data.
From current data, select one industry or category, then query historical data for more info.
'''
START_YEAR=2003 #full record keeping starts in 2003, data incomplete before then
MAX_YEARS=25
M_IN_Y=12 #12 month in a year
END_YEAR=START_YEAR+MAX_YEARS-1
LIMIT=70000
SLEEP_TIME=10
#'text':'code description',
#'count':0 #number of license in this code
str_type='application_type'
str_lnum='license_number'
str_issue='date_issued'
str_start='license_start_date'
str_end='expiration_date'
str_lcode='license_code'
str_l_des='license_description'
str_b_id='business_activity_id'
str_bid_des='business_activity'
unknown='UNKNOWN'

BUS_LIC='r5kz-chrr'

def getindex(year,month):
    '''Find index position in counter from text string year and month
    '''    
    #If no input assume earlier than 2003 when record keeping is poor
    if year == '':
        return 0;
    i_year = int(year)
    if i_year < START_YEAR:
        index=0
    elif i_year > END_YEAR:
        index = MAX_YEARS*M_IN_Y-1
    else:    
        index=(i_year-START_YEAR)*M_IN_Y+int(month)-1
    return index

class business_lic:
    '''Class for handling business license data from 2003
    to present.
    '''
    
    def __init__(self,mongo_collection):
        '''After creating object, either call load() to get previous saved data,
        or pull_hist_lic() to obtain data
        '''        
        self.lcode_dict={}
        self.b_id_dict={}
        self.collection=mongo_collection
                    
    #function is slow due to rate limiting from server
    def pull_hist_lic(self):
        '''pulling business license history from chicago open data portal.
        to initialize data 
        '''
    #https://data.cityofchicago.org/resource/r5kz-chrr.json
    # Unauthenticated client only works with public data sets. Note 'None'
    # in place of application token, and no username or password:
    # Example authenticated client (needed for non-public datasets):
    # client = Socrata(data.cityofchicago.org,
    #                  MyAppToken,
    #                  userame="user@example.com",
    #                  password="AFakePassword")    
    
        client = Socrata("data.cityofchicago.org", None)   
        
        #A list of year/month where we limit the search to license issued in that month 
        date_list=[ '20'+str(i).zfill(2)+'-'+str(j).zfill(2) for i in range(3,22) for j in range(1,13) ]
        
        db_offset=0
        for i in range(240):
            search_term = '( {} > {} ) AND ({} < {})'.format(str_issue,date_list[i],
                                                             str_issue,date_list[i+1])
            retrived = LIMIT
            
            while retrived == LIMIT:
                results=client.get(BUS_LIC,limit=LIMIT,order='id',where=search_term,offset=db_offset)
                retrived=len(results)
                db_offset += retrived    
                self.collection.insert_many(results)
                print(f" {db_offset} records have been inserted.")
                time.sleep(SLEEP_TIME)
                
        self.collection.create_index([(str_lcode, pymongo.ASCENDING),(str_lnum, pymongo.ASCENDING)])     
        self.compact()
        
    def compact(self):            
#        pipeline=[{'$group':{'_id':{'lcode':'$license_code', 'lnum':'$license_number',
#           'acct_num':'$account_number','site':'$site_number'},'city':{'$first':'$city'},
#            str_start:{'$min':'$license_start_date'}, str_end:{'$max': '$expiration_date'},
#            str_issue:{'$min':'$date_issued'}, 'address':{'$first':'$address'},
#            's':{'$first':'$state'},'zip':{'$first':'$zip_code'}}},
#            {'$match':{'city':'CHICAGO'}}]
        pipeline=[{'$match':{'city':'CHICAGO'}},{'$group':{'_id':{'lcode':'$license_code', 'lnum':'$license_number'},
            str_issue:{'$min':'$date_issued'},str_start:{'$min':'$license_start_date'}, str_end:{'$max':'$expiration_date'},
            'b_id':{'$max':'$business_activity_id'}}}, {'$project':{'_id':0, str_lcode:'$_id.lcode',str_lnum:'$_id.lnum'
            ,str_b_id:'$b_id',str_issue:1,str_start:1,str_end:1}}]
        cursor=list(self.collection.aggregate(pipeline,allowDiskUse=True)) 
        self.df_compat=pd.DataFrame(cursor)
        self.df_compat[str_b_id]=self.df_compat[str_b_id].apply(lambda x: '-1' if pd.isnull(x) else x)
        
#        cursor=lic_history.find({'city':'CHICAGO'},{'@_id':0})
        self.df_compat.set_index([str_lcode,str_lnum],inplace=True)
        self.df_compat.sort_index(inplace=True)
        cursor=self.collection.find({'city':'CHICAGO'},{'@_id':0})
        self.make_dict(cursor)       
        self.save()
        
    def save(self):
        '''save class data into pickle files
        '''
        pickle.dump(self.lcode_dict,open('lic_code.p',"wb"))
        pickle.dump(self.b_id_dict,open("b_id_code.p","wb"))
        pickle.dump(self.df_compat,open("df.p","wb"))
        
    def load(self):
        '''load class data from pickle files
        '''
        self.lcode_dict=pickle.load(open('lic_code.p','rb'))
        self.b_id_dict=pickle.load(open("b_id_code.p","rb"))
        self.df_compat=pickle.load(open("df.p","rb"))
       
    def make_dict(self,cursor):
        '''read from mongoDB to create dictionary containing license counters
        '''       
        for i in cursor:
            #count the business in every month the license is valid    
            if str_lcode in i:    
                lcode=i[str_lcode]
                if lcode in self.lcode_dict:
                    self.lcode_dict[lcode]['count']+=1
                    self.lcode_dict[lcode]['lnum'].add(i[str_lnum])
                else:
                    #create counters
                    if str_l_des in i:
                        text=i[str_l_des]
                    else:
                        text=unknown
                
                    self.lcode_dict[lcode]={'text':text,'count':1,'lnum':{i[str_lnum]}}

            text='Unspecified'
            if str_b_id in i:
                #Several b_ids may exist in this field.                
                b_ids=i[str_b_id].replace(' ','').split('|')                
            else:
                #missing business_activity_id will be treated as -1 
                b_ids=['-1']
                
            for index, b_id in enumerate(b_ids):
                    if b_id in self.b_id_dict:
                        self.b_id_dict[b_id]['count']+=1
                        self.b_id_dict[b_id]['lnum'].add(i[str_lnum])
                    else:
                        #create counters
                
                        #if multiple business activity id, use
                        #text in business activity mstching id.
                        if (str_bid_des in i):
                            text=i[str_bid_des].split('|')[index]                    
                                            
                        self.b_id_dict[b_id]={'text':text,'count':1,'lnum':{i[str_lnum]}}            
                        
    def lsummary(self):                            
        #create from dict summary table
        total=0
        for i in range(0,9999):
            lcode=str(i)
            if lcode in self.lcode_dict:
                count=self.lcode_dict[lcode]['count']
                print(i, count, len(self.lcode_dict[lcode]['lnum']), self.lcode_dict[lcode]['text'])
                total+=count
        print(total)
        
    def bsummary(self):
        totalb=0
        for i in range(-1,2000):
            b_id=str(i)
            if b_id in self.b_id_dict:
                count = self.b_id_dict[b_id]['count']
                print(i, count, len(self.b_id_dict[b_id]['lnum']),self.b_id_dict[b_id]['text'])
                totalb+=count
        print(totalb)
        
    def update(self):
        '''pull latest and update class data
        '''    
        client = Socrata("data.cityofchicago.org", None)    
        retrived = LIMIT
        db_offset=0
        end_date='2021-07'
        search_term = '( {} > {} )'.format(str_issue,end_date)
        
        while retrived == LIMIT:
            results=client.get(BUS_LIC,limit=LIMIT,where=search_term, order='id', offset=db_offset)
            retrived=len(results)
            db_offset += retrived
#            self.collection.insert_many(results)
            make_dict(results)
            print(f" {db_offset} records have been inserted.")
            time.sleep(SLEEP_TIME)
        self.refresh=date.today()    
        save()
        
        
CUR_BUS_LIC='uupf-x98q'

class active_lic:  
    def __init__(self):
        self.refresh_date=date.today()
#        self.data=pickle.load(open('current.p','rb'))
#        self.generate()


    def save(self):
#        self.data.to_csv('data/self.data')
        pickle.dump(self.data,open('current.p','wb'))
        pickle.dump(self.lcode_dict,open('a_code.p','wb'))
        pickle.dump(self.b_id_dict,open('a_bid.p','wb'))    
    def load(self):
#        self.data=pd.read_csv('data/active.csv')
        self.data=pickle.load(open('current.p','rb'))
        self.lcode_dict=pickle.load(open('a_code.p','rb'))
        self.b_id_dict=pickle.load(open('a_bid.p','rb'))
            
    def pull_curr_lic(self):
        '''pulling active business license from chicago open data portal.
        '''
        #https://data.cityofchicago.org/resource/uupf-x98q.json    
        time_delta=date.today()-self.refresh_date
        if time_delta.days == 0:
            return
        self.refresh_date=date.today()    
        client = Socrata("data.cityofchicago.org", None) 
        results=client.get(CUR_BUS_LIC,limit=LIMIT,order='id',city='CHICAGO')
        retrived=len(results)
        # Convert to pandas DataFrame
        df = pd.DataFrame.from_records(results)
        db_offset=retrived
        while retrived == LIMIT:
            print(f" {db_offset} records have been processed.")
            time.sleep(SLEEP_TIME)
            results=client.get(CUR_BUS_LIC,limit=LIMIT,order='id', offset=db_offset)
            retrived=len(results)
            db_offset += retrived   
            results_df = pd.DataFrame.from_records(results)
            df=pd.concat([df,results_df]) 
                                   
        #non-chicago data sometimes leak in            
        data=df[df['city']=='CHICAGO']
        data=data.reset_index(drop=True,inplace=True) 
        #missing business_activity_id will be treated as -1                   
        df[str_b_id].fillna(value='-1',inplace=True)
        df[str_bid_des].fillna(value='Unspecified',inplace=True)
        #For missing GPS data, set it to chicago.
        df['longitude'].fillna(value=-87.623177,inplace=True)
        df['latitude'].fillna(value=41.881832,inplace=True)

        to_str=lambda x: str(x)
        data[str_lcode]=data[str_lcode].apply(to_str)
        data[str_lnum]=data[str_lnum].apply(to_str)
        
        data.set_index([str_lcode,str_lnum],inplace=True)
        data.sort_index(inplace=True)
        self.data=data
        self.generate()
        
    def generate(self):
#        self.lcode_dict={'ALL':{'text':'ALL active licenses','count':len(self.data)}}
        self.lcode_dict={}
        self.b_id_dict={}
        
        for index,i in self.data.iterrows():
            #count the business in every month the license is valid
            #index[0] is license code, index[1] is license number
            
            if index[0] in self.lcode_dict:
                self.lcode_dict[index[0]]['count']+=1
                
#                self.lcode_dict[index[0]]['lnum'].add(index[1])
            else:
                #create counters
                text=i[str_l_des]
                
                self.lcode_dict[index[0]]={'text':text,'count':1,'lnum':{index[1]}}

            #Several b_ids may exist in this field.                
            b_ids=i[str_b_id].replace(' ','').split('|')                
                
            for index, b_id in enumerate(b_ids):
                if b_id in self.b_id_dict:
                    self.b_id_dict[b_id]['count']+=1
                    self.b_id_dict[b_id]['lnum'].add(index)
                else:
                    #create counters
                
                        #if multiple business activity id, use
                        #text in business activity mstching id.
                    if i[str_bid_des]:
                        text=i[str_bid_des].split('|')[index]                    
                                            
                    self.b_id_dict[b_id]={'text':text,'count':1,'lnum':{index}}
        self.save()
        
    def find_code(self,search_code,old,column_name,all_code=False):
        '''Give a summary for the given license code search_code is str,
        column_name only support 'licence_code' and 'business_activity_id'
        '''
        #from current license dataframe, get a list of license number whose 
        #license code matches the given license code 
        output=''
        return_list=[]
        if all_code :
#            data_list=set(self.data[str_lnum])
            self.mapdata=self.data
            df_lcode=old.df_compat
        else:    
#            data_list = set(self.data[ self.data[str_lcode] == int(lcode)] [str_lnum])

            if column_name == str_lcode:
                #multi-index, just select the right slice
                self.mapdata=self.data.loc[ (search_code,) ]
                df_lcode=old.df_compat.loc[ (search_code,) ]
            else:
            #for business_activity_id, both are str format. And multiple bid are possible in dataframe 
                list1=[ search_code in i[column_name] for idx,i in self.data.iterrows() ]
                self.mapdata=self.data[ list1 ]
                
                list2=[ search_code in i[column_name] for idx,i in old.df_compat.iterrows() ]
                df_lcode=old.df_compat[ list2 ]
                
        
        curr_lic=len(self.mapdata)            
        total_lic = len(df_lcode)    
        now_=datetime.now()
        today_index=getindex(str(now_.year),str(now_.month))
        month_in_business = 0        

        license_counter=np.zeros((MAX_YEARS*M_IN_Y))

        for index,i in df_lcode.iterrows():
        #search the historial business_lic database to find the license duration for 
        #each license matching the license code that is still active today. 
        #The same license number may be used by two different users if license codes are 
        #different. But for the same license code, the license number is unique per business.
        
            #using the earliest license start date for each business, 
            #sum the time in business for all business still active now. 
                if i[str_start]:
                    start=i[str_start]
                else:
                    start=i[str_issue]
                start_index=getindex(start[0:4], start[5:7])
                
                if i[str_end]:
                    end=i[str_end]
                else:
                    end=start
                end_index=getindex(end[0:4], end[5:7])
                
                #likely typo of wrong year, 
                if end_index < start_index:
                    end_index=start_index+12
                    
                license_counter[start_index:end_index+1]+=1                  
                    
                #active license should have expiration date 
                #after today 
                
                if now_ < pd.to_datetime(end):                 
                    month_in_business += (end_index+1-start_index)
                    
                    #check if shall be included in sample list. Items in sample list has earliest
                    #license_start_date
                    
                    
        #Get a few examples. Convert so key are the same type to merge
        self.mapdata = self.mapdata.join(df_lcode,rsuffix='_y')
        self.outdf=self.mapdata.sort_values(by=['license_start_date_y'])
        self.outdf=self.mapdata.head(10)
        
        if all_code:
            output+='All active license codes.\n' 
        elif column_name == str_lcode:    
            output+='License code {} : {}\n'.format(search_code,old.lcode_dict[search_code]['text'])
        else:
            output+='business activity {} : {}\n'.format(search_code,old.b_id_dict[search_code]['text'])
        #record keeping is spotty before 2003, so time in business earlier than
        #2003 may not be included
        
        total=sum(license_counter[0:today_index+1])
        total_active = month_in_business 
        total_non_active = total - total_active
        
        non_active_lic = total_lic - curr_lic
        
        average_act = total_active
        
        average=total/total_lic
        
        if curr_lic == 0:
            average_act = 0
        else:
            average_act = total_active/curr_lic
            
        if non_active_lic == 0:
            average_nonact = 0
        else:
            average_nonact = total_non_active/non_active_lic
        
        bar,ax=plt.subplots(1,2,figsize=[16,8])
        plt.subplot(1,2,1)
        tick_list=['active business','all','non active business']
        plt.bar(range(3),[average_act, average, average_nonact],tick_label=tick_list)
        plt.title('Average length in business')
        
        plt.subplot(1,2,2)
        plt.bar(range(3),[curr_lic, total_lic, non_active_lic],tick_label=tick_list)
        plt.title('Number of business')
        
        
        fig,ax2=plt.subplots(1,2,figsize=[16,8])
        plt.subplot(1,2,1)
        plt.plot(license_counter[0:today_index+7])
        plt.plot([today_index,today_index],[0,license_counter[today_index]],linestyle='-')
        plt.grid()
        ticks_x=np.arange(0,240,36)
        x_labels=['2003', '2006', '2009','2012','2015','2018','2021']
        plt.xticks(ticks_x, x_labels, fontsize = '20', color='green',rotation=45);   
        plt.title('Number of active business over the years')
        
        plt.subplot(1,2,2)
        ticks_x = np.arange(-2, 7)
        plt.plot(ticks_x,license_counter[today_index-2:today_index+7])
        plt.plot([0,0],[0,license_counter[today_index]],linestyle=':')
        plt.grid()
        more_text=[ str(i)+'mon. adv.' for i in range(1,7) ]
        x_labels=['2 mon. prior', '1 mon. prior', 'CURRENT']+more_text
        plt.xticks(ticks_x, x_labels, fontsize = '20', color='orange',rotation=90);
        plt.title('Number of license still active in the future')         
        return fig,bar,output
    
