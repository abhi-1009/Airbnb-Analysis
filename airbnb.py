import pandas as pd
import numpy as np
import streamlit as st
import json
from streamlit_option_menu import option_menu
pd.set_option('display.max_columns', None)
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from PIL import Image

# Path to the JSON file
json_file_path = r'C:/Users/Hp/OneDrive/Desktop/python/sample_airbnb.json'

# Read the JSON file into a DataFrame
df = pd.read_json(json_file_path)

# Convert the DataFrame to CSV
csv_file_path = r'C:/Users/Hp/OneDrive/Desktop/python/sample_airbnb.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file has been saved at {csv_file_path}")

sample_airbnb = 'C:/Users/Hp/OneDrive/Desktop/python/sample_airbnb.csv'
airbnb = pd.read_csv(sample_airbnb)

# dropping non values data
airbnb.drop(['neighborhood_overview', 'last_scraped', 'review_scores', 'first_review','description','reviews_per_month','monthly_price','weekly_price','summary', 'space', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'calendar_last_scraped', 'last_review', 'security_deposit'], axis=1, inplace= True)

airbnb["beds"].fillna(0,inplace= True)
airbnb["bedrooms"].fillna(0,inplace= True)
airbnb["bathrooms"].fillna(0,inplace= True)
airbnb["cleaning_fee"].fillna(0,inplace= True)

airbnb["minimum_nights"]= airbnb["minimum_nights"].astype(int)
airbnb["maximum_nights"]= airbnb["maximum_nights"].astype(int)
airbnb["price"]= airbnb["price"].astype(str).astype(float).astype(int)
airbnb["bedrooms"]= airbnb["bedrooms"].astype(int)
airbnb["beds"]= airbnb["beds"].astype(int)
airbnb["bathrooms"]= airbnb["bathrooms"].astype(str).astype(float).astype(int)
airbnb["extra_people"]= airbnb["extra_people"].astype(str).astype(float).astype(int)
airbnb["guests_included"]= airbnb["guests_included"].astype(str).astype(float).astype(int)
airbnb["cleaning_fee"]= airbnb["cleaning_fee"].astype(str).astype(float).astype(int)

# Extracting all values from the 'host' column
hosts = airbnb['host'].tolist()

import ast

# Convert the list of dictionary strings to actual dictionaries
hosts = airbnb['host'].apply(ast.literal_eval)

# Convert the list of dictionaries to a DataFrame
host_details = pd.DataFrame(hosts.tolist())

# Set display options
pd.set_option('display.max_rows', 5555)  # Display all rows
pd.set_option('display.max_columns', 16)  # Display all columns
pd.set_option('display.width', None)  # Allow wide tables to wrap
warnings.filterwarnings("ignore")

# host_neighbourhood have more empty values ('')
# Finding the how many values are empty
list_index= []
for index,row in host_details.iterrows():
    if row["host_neighbourhood"] =='':
        list_index.append(index)
        
host_details["host_response_time"].fillna("Not Specified",inplace= True)
host_details["host_response_rate"].fillna("Not Specified",inplace= True)
host_details["host_neighbourhood"]= host_details["host_neighbourhood"].replace({'':"Not Specified"})

# Changing the "True" or "False" features

host_details["host_is_superhost"]= host_details["host_is_superhost"].map({False: "No", True: "Yes"})
host_details["host_has_profile_pic"]= host_details["host_has_profile_pic"].map({False: "No", True: "Yes"})
host_details["host_identity_verified"]= host_details["host_identity_verified"].map({False: "No", True: "Yes"})

# Extract the 'address' column and convert it to a list
address_data = airbnb['address'].tolist()

# Convert the list of dictionary strings to actual dictionaries
addresses = airbnb['address'].apply(ast.literal_eval)

# Convert the list of dictionaries to a DataFrame
address_details = pd.DataFrame(addresses.tolist())

# Set display options
pd.set_option('display.max_rows', 5555)  # Display all rows
pd.set_option('display.max_columns', 9)  # Display all columns
pd.set_option('display.width', None)  # Allow wide tables to wrap
warnings.filterwarnings("ignore")

# Extract the 'location' column
locations = address_details['location'].apply(lambda x: x if isinstance(x, dict) else {})

# Convert the list of dictionaries to a DataFrame
location_details = pd.DataFrame(locations.tolist())

# Set display options
pd.set_option('display.max_rows', 5555)  # Display all rows
pd.set_option('display.max_columns', 4)  # Display all columns
pd.set_option('display.width', None)  # Allow wide tables to wrap
warnings.filterwarnings("ignore")

# Ensure display options for pandas
pd.set_option('display.max_rows', 5555)  # Display all rows
pd.set_option('display.max_columns', 4)  # Display all columns
pd.set_option('display.width', None)  # Allow wide tables to wrap
warnings.filterwarnings("ignore")

# Function to ensure the location data is in dictionary form
def ensure_dict(location):
    if isinstance(location, str):
        try:
            location = json.loads(location.replace("'", "\""))  # Adjust for single quotes if necessary
        except json.JSONDecodeError:
            return {}
    return location if isinstance(location, dict) else {}

# Apply the function to the 'location' column
locations = address_details['location'].apply(ensure_dict)

# Function to extract coordinates and convert them to numeric
def extract_coordinates(coord_dict):
    if 'coordinates' in coord_dict:
        coords = coord_dict['coordinates']
        if isinstance(coords, list) and len(coords) == 2:
            longitude, latitude = coords  # Note the order of longitude and latitude
            return pd.Series({
                'latitude': pd.to_numeric(latitude, errors='coerce'),
                'longitude': pd.to_numeric(longitude, errors='coerce')
            })
    return pd.Series({'latitude': None, 'longitude': None})

# Apply the extraction function to each element in the Series
coordinates_df = locations.apply(extract_coordinates)

# Extracting all values from the 'availability' column

availability_data = airbnb['availability'].tolist()

# Convert the list of dictionary strings to actual dictionaries
availabilities = airbnb['availability'].apply(ast.literal_eval)

# Convert the list of dictionaries to a DataFrame
availability_details = pd.DataFrame(availabilities.tolist())

# Set display options
pd.set_option('display.max_rows', 5555)  # Display all rows
pd.set_option('display.max_columns', 4)  # Display all columns
pd.set_option('display.width', None)  # Allow wide tables to wrap
warnings.filterwarnings("ignore")

# Extracting all values from the 'amenities' column
amenities = airbnb['amenities'].tolist()

amenities = airbnb['amenities'].apply(ast.literal_eval)

# Convert the list of lists to a DataFrame
amenities_details = pd.DataFrame({'_id': airbnb['_id'], 'amenities': amenities})

def sort_amenities(x):
    x.sort()
    return x
 
amenities_details['amenities'] = amenities_details['amenities'].apply(sort_amenities)

# Set display options
pd.set_option('display.max_rows', 5555)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # Allow wide tables to wrap
warnings.filterwarnings("ignore")

# Merging all DataFrames

airbnb_details = airbnb.drop(columns=['host', 'address', 'availability', 'amenities'])
airbnb_details = airbnb_details.join(host_details, rsuffix='_host')
airbnb_details = airbnb_details.join(address_details, rsuffix='_address')
airbnb_details = airbnb_details.join(availability_details, rsuffix='_availability')
airbnb_details = airbnb_details.join(amenities_details.set_index('_id'), on='_id')
airbnb_details = airbnb_details.join(location_details, rsuffix='_location')
airbnb_details = airbnb_details.join(coordinates_df, rsuffix='_coordinates')

# Save the merged DataFrame to CSV

merged_csv_path = r'C:/Users/Hp/OneDrive/Desktop/python/Airbnb.csv'
airbnb_details.to_csv(merged_csv_path, index=False)
print(f"Merged CSV file has been saved at {merged_csv_path}")
airbnb_details= pd.read_csv("C:/Users/Hp/OneDrive/Desktop/python/Airbnb.csv")

# Streamlit part

st.set_page_config(layout="wide")

st.title("AIRBNB DATA ANALYSIS")
st.write("")

def datafr():
    df = pd.read_csv("C:/Users/Hp/OneDrive/Desktop/python/Airbnb.csv")    
    return df
	
df = datafr()

with st.sidebar:
    select= option_menu("Main Menu", ["Home", "About", "Data Exploration"])

if select == "Home":

    image1= Image.open("C:/Users/Hp/OneDrive/Desktop/python/images airbnb.jfif")
    st.image(image1)

    st.header("About Airbnb")
    st.write("")
    st.write('''***Airbnb is an online marketplace that connects people who want to rent out
              their property with people who are looking for accommodations,
              typically for short stays. Airbnb offers hosts a relatively easy way to
              earn some income from their property.Guests often find that Airbnb rentals
              are cheaper and homier than hotels.***''')
    st.write("")
    st.write('''***Airbnb Inc (Airbnb) operates an online platform for hospitality services.
                  The company provides a mobile application (app) that enables users to list,
                  discover, and book unique accommodations across the world.
                  The app allows hosts to list their properties for lease,
                  and enables guests to rent or lease on a short-term basis,
                  which includes vacation rentals, apartment rentals, homestays, castles,
                  tree houses and hotel rooms. The company has presence in China, India, Japan,
                  Australia, Canada, Austria, Germany, Switzerland, Belgium, Denmark, France, Italy,
                  Norway, Portugal, Russia, Spain, Sweden, the UK, and others.
                  Airbnb is headquartered in San Francisco, California, the US.***''')
    
    st.header("Background of Airbnb")
    st.write("")
    st.write('''***Airbnb was born in 2007 when two Hosts welcomed three guests to their
              San Francisco home, and has since grown to over 4 million Hosts who have
                welcomed over 1.5 billion guest arrivals in almost every country across the globe.***''')
if select == "Data Exploration":
    tab1, tab2, tab3, tab4, tab5= st.tabs(["***PRICE ANALYSIS***","***AVAILABILITY ANALYSIS***","***LOCATION ANALYSIS***", "***GEOSPATIAL VISUALIZATION***", "***TOP CHARTS***"])

    with tab1:
        st.title("**PRICE ANALYSIS**")
        col1,col2= st.columns(2)

    with col1:
            
            country= st.selectbox("Select the Country",df["country"].unique())

            df1= df[df["country"] == country]
            df1.reset_index(drop= True, inplace= True)

            room_ty= st.selectbox("Select the Room Type",df1["room_type"].unique())
            
            df2= df1[df1["room_type"] == room_ty]
            df2.reset_index(drop= True, inplace= True)

            df_bar= pd.DataFrame(df2.groupby("property_type")[["price","number_of_reviews"]].sum())
            df_bar.reset_index(inplace= True)

            fig_bar= px.bar(df_bar, x='property_type', y= "price", title= "PRICE FOR PROPERTY_TYPES",hover_data=["number_of_reviews"],color_discrete_sequence=px.colors.sequential.Redor_r, width=600, height=500)
            st.plotly_chart(fig_bar)
     
    with col2:
            
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
     
            proper_ty= st.selectbox("Select the Property_type",df2["property_type"].unique())

            df4= df2[df2["property_type"] == proper_ty]
            df4.reset_index(drop= True, inplace= True)

            df_pie= pd.DataFrame(df4.groupby("host_response_time")[["price","bedrooms"]].sum())
            df_pie.reset_index(inplace= True)

            fig_pi= px.pie(df_pie, values="price", names= "host_response_time",
                            hover_data=["bedrooms"],
                            color_discrete_sequence=px.colors.sequential.BuPu_r,
                            title="PRICE DIFFERENCE BASED ON HOST RESPONSE TIME",
                            width= 600, height= 500)
            st.plotly_chart(fig_pi)

    col1,col2= st.columns(2)

    with col1:
            
            hostresponsetime= st.selectbox("Select the host_response_time",df4["host_response_time"].unique())

            df5= df4[df4["host_response_time"] == hostresponsetime]

            df_do_bar= pd.DataFrame(df5.groupby("bed_type")[["minimum_nights","maximum_nights","price"]].sum())
            df_do_bar.reset_index(inplace= True)

            fig_do_bar = px.bar(df_do_bar, x='bed_type', y=['minimum_nights', 'maximum_nights'], 
            title='MINIMUM NIGHTS AND MAXIMUM NIGHTS',hover_data="price",
            barmode='group',color_discrete_sequence=px.colors.sequential.Rainbow, width=600, height=500)
            
            st.plotly_chart(fig_do_bar)

    with col2:

            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")

            df_do_bar_2= pd.DataFrame(df5.groupby("bed_type")[["bedrooms","beds","accommodates","price"]].sum())
            df_do_bar_2.reset_index(inplace= True)

            fig_do_bar_2 = px.bar(df_do_bar_2, x='bed_type', y=['bedrooms', 'beds', 'accommodates'], 
            title='BEDROOMS AND BEDS ACCOMMODATES',hover_data="price",
            barmode='group',color_discrete_sequence=px.colors.sequential.Rainbow_r, width= 600, height= 500)
           
            st.plotly_chart(fig_do_bar_2)
            
    with tab2:

        def datafr():
            df_a= pd.read_csv("C:/Users/Hp/OneDrive/Desktop/python/Airbnb.csv")
            return df_a

        df_a= datafr()

        st.title("**AVAILABILITY ANALYSIS**")
        col1,col2= st.columns(2)

        with col1:
            
            country_a= st.selectbox("Select the Country_a",df_a["country"].unique())

            df1_a= df[df["country"] == country_a]
            df1_a.reset_index(drop= True, inplace= True)

            property_ty_a= st.selectbox("Select the Property Type",df1_a["property_type"].unique())
            
            df2_a= df1_a[df1_a["property_type"] == property_ty_a]
            df2_a.reset_index(drop= True, inplace= True)

            df_a_sunb_30= px.sunburst(df2_a, path=["room_type","bed_type","is_location_exact"], values="availability_30",width=600,height=500,title="Availability_30",color_discrete_sequence=px.colors.sequential.Peach_r)
            st.plotly_chart(df_a_sunb_30)
        
        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            
            df_a_sunb_60= px.sunburst(df2_a, path=["room_type","bed_type","is_location_exact"], values="availability_60",width=600,height=500,title="Availability_60",color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(df_a_sunb_60)

        col1,col2= st.columns(2)

        with col1:
            
            df_a_sunb_90= px.sunburst(df2_a, path=["room_type","bed_type","is_location_exact"], values="availability_90",width=600,height=500,title="Availability_90",color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
            st.plotly_chart(df_a_sunb_90)

        with col2:

            df_a_sunb_365= px.sunburst(df2_a, path=["room_type","bed_type","is_location_exact"], values="availability_365",width=600,height=500,title="Availability_365",color_discrete_sequence=px.colors.sequential.Greens_r)
            st.plotly_chart(df_a_sunb_365)
        
        roomtype_a= st.selectbox("Select the Room Type_a", df2_a["room_type"].unique())

        df3_a= df2_a[df2_a["room_type"] == roomtype_a]

        df_mul_bar_a= pd.DataFrame(df3_a.groupby("host_response_time")[["availability_30","availability_60","availability_90","availability_365","price"]].sum())
        df_mul_bar_a.reset_index(inplace= True)

        fig_df_mul_bar_a = px.bar(df_mul_bar_a, x='host_response_time', y=['availability_30', 'availability_60', 'availability_90', "availability_365"], 
        title='AVAILABILITY BASED ON HOST RESPONSE TIME',hover_data="price",
        barmode='group',color_discrete_sequence=px.colors.sequential.Rainbow_r,width=1000)

        st.plotly_chart(fig_df_mul_bar_a)

    with tab3:

        st.title("LOCATION ANALYSIS")
        st.write("")

        def datafr():
            df= pd.read_csv("C:/Users/Hp/OneDrive/Desktop/python/Airbnb.csv")
            return df

        df_l= datafr()

        country_l= st.selectbox("Select the Country_l",df_l["country"].unique())

        df1_l= df_l[df_l["country"] == country_l]
        df1_l.reset_index(drop= True, inplace= True)

        proper_ty_l= st.selectbox("Select the Property_type_l",df1_l["property_type"].unique())

        df2_l= df1_l[df1_l["property_type"] == proper_ty_l]
        df2_l.reset_index(drop= True, inplace= True)

        st.write("")

        def select_the_df(sel_val):
            if sel_val == str(df2_l['price'].min())+' '+str('to')+' '+str(differ_max_min*0.30 + df2_l['price'].min())+' '+str("(30% of the Value)"):

                df_val_30= df2_l[df2_l["price"] <= differ_max_min*0.30 + df2_l['price'].min()]
                df_val_30.reset_index(drop= True, inplace= True)
                return df_val_30

            elif sel_val == str(differ_max_min*0.30 + df2_l['price'].min())+' '+str('to')+' '+str(differ_max_min*0.60 + df2_l['price'].min())+' '+str("(30% to 60% of the Value)"):
            
                df_val_60= df2_l[df2_l["price"] >= differ_max_min*0.30 + df2_l['price'].min()]
                df_val_60_1= df_val_60[df_val_60["price"] <= differ_max_min*0.60 + df2_l['price'].min()]
                df_val_60_1.reset_index(drop= True, inplace= True)
                return df_val_60_1
            
            elif sel_val == str(differ_max_min*0.60 + df2_l['price'].min())+' '+str('to')+' '+str(df2_l['price'].max())+' '+str("(60% to 100% of the Value)"):

                df_val_100= df2_l[df2_l["price"] >= differ_max_min*0.60 + df2_l['price'].min()]
                df_val_100.reset_index(drop= True, inplace= True)
                return df_val_100
            
        differ_max_min= df2_l['price'].max()-df2_l['price'].min()

        val_sel= st.radio("Select the Price Range",[str(df2_l['price'].min())+' '+str('to')+' '+str(differ_max_min*0.30 + df2_l['price'].min())+' '+str("(30% of the Value)"),
                                                    
                                                    str(differ_max_min*0.30 + df2_l['price'].min())+' '+str('to')+' '+str(differ_max_min*0.60 + df2_l['price'].min())+' '+str("(30% to 60% of the Value)"),

                                                    str(differ_max_min*0.60 + df2_l['price'].min())+' '+str('to')+' '+str(df2_l['price'].max())+' '+str("(60% to 100% of the Value)")])
                                          
        df_val_sel= select_the_df(val_sel)

        st.dataframe(df_val_sel)

        df_val_sel_gr= pd.DataFrame(df_val_sel.groupby("accommodates")[["cleaning_fee","bedrooms","beds","extra_people"]].sum())
        df_val_sel_gr.reset_index(inplace= True)

        fig_1= px.bar(df_val_sel_gr, x="accommodates", y= ["cleaning_fee","bedrooms","beds"], title="ACCOMMODATES",
                    hover_data= "extra_people", barmode='group', color_discrete_sequence=px.colors.sequential.Rainbow_r,width=1000)
        st.plotly_chart(fig_1)
        
        room_ty_l= st.selectbox("Select the Room_Type_l", df_val_sel["room_type"].unique())

        df_val_sel_rt= df_val_sel[df_val_sel["room_type"] == room_ty_l]

        fig_2= px.bar(df_val_sel_rt, x= ["street","host_location","host_neighbourhood"],y="market", title="MARKET",
                    hover_data= ["name","host_name","market"], barmode='group',orientation='h', color_discrete_sequence=px.colors.sequential.Rainbow_r,width=1000)
        st.plotly_chart(fig_2)

        fig_3= px.bar(df_val_sel_rt, x="government_area", y= ["host_is_superhost","host_neighbourhood","cancellation_policy"], title="GOVERNMENT_AREA",
                    hover_data= ["guests_included","type"], barmode='group', color_discrete_sequence=px.colors.sequential.Rainbow_r,width=1000)
        st.plotly_chart(fig_3)  

    with tab4:

        st.title("GEOSPATIAL VISUALIZATION")
        st.write("")
        fig_4 = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='price', size='accommodates',
                        color_continuous_scale= "rainbow",hover_name='name',range_color=(0,49000), 
                        mapbox_style="carto-positron",
                        zoom=1)
        fig_4.update_layout(width=1150,height=800,title='Geospatial Distribution of Listings')
        st.plotly_chart(fig_4)
    
    with tab5:
        st.title("**TOP CHARTS**")

        country_t= st.selectbox("Select the Country_t",df["country"].unique())

        df1_t= df[df["country"] == country_t]

        property_ty_t= st.selectbox("Select the Property_type_t",df1_t["property_type"].unique())

        df2_t= df1_t[df1_t["property_type"] == property_ty_t]
        df2_t.reset_index(drop= True, inplace= True)

        df2_t_sorted= df2_t.sort_values(by="price")
        df2_t_sorted.reset_index(drop= True, inplace= True)

        df_price= pd.DataFrame(df2_t_sorted.groupby("host_neighbourhood")["price"].agg(["sum","mean"]))
        df_price.reset_index(inplace= True)
        df_price.columns= ["host_neighbourhood", "Total_price", "Avarage_price"]
        
        col1, col2= st.columns(2)

        with col1:
            
            fig_price= px.bar(df_price, x= "Total_price", y= "host_neighbourhood", orientation='h',
                            title= "PRICE BASED ON HOST_NEIGHBOURHOOD", width= 600, height= 800)
            st.plotly_chart(fig_price)

        with col2:

            fig_price_2= px.bar(df_price, x= "Avarage_price", y= "host_neighbourhood", orientation='h',
                                title= "AVERAGE PRICE BASED ON HOST_NEIGHBOURHOOD",width= 600, height= 800)
            st.plotly_chart(fig_price_2)

        col1, col2= st.columns(2)

        with col1:

            df_price_1= pd.DataFrame(df2_t_sorted.groupby("host_location")["price"].agg(["sum","mean"]))
            df_price_1.reset_index(inplace= True)
            df_price_1.columns= ["host_location", "Total_price", "Avarage_price"]
            
            fig_price_3= px.bar(df_price_1, x= "Total_price", y= "host_location", orientation='h',
                                width= 600,height= 800,color_discrete_sequence=px.colors.sequential.Bluered_r,
                                title= "PRICE BASED ON HOST_LOCATION")
            st.plotly_chart(fig_price_3)

        with col2:

            fig_price_4= px.bar(df_price_1, x= "Avarage_price", y= "host_location", orientation='h',
                                width= 600, height= 800,color_discrete_sequence=px.colors.sequential.Bluered_r,
                                title= "AVERAGE PRICE BASED ON HOST_LOCATION")
            st.plotly_chart(fig_price_4)

        room_type_t= st.selectbox("Select the Room_Type_t",df2_t_sorted["room_type"].unique())

        df3_t= df2_t_sorted[df2_t_sorted["room_type"] == room_type_t]

        df3_t_sorted_price= df3_t.sort_values(by= "price")

        df3_t_sorted_price.reset_index(drop= True, inplace = True)

        df3_top_50_price= df3_t_sorted_price.head(100)

        fig_top_50_price_1= px.bar(df3_top_50_price, x= "name",  y= "price" ,color= "price",
                                 color_continuous_scale= "rainbow",
                                range_color=(0,df3_top_50_price["price"].max()),
                                title= "MINIMUM_NIGHTS MAXIMUM_NIGHTS AND ACCOMMODATES",
                                width=1200, height= 800,
                                hover_data= ["minimum_nights","maximum_nights","accommodates"])
        
        st.plotly_chart(fig_top_50_price_1)

        fig_top_50_price_2= px.bar(df3_top_50_price, x= "name",  y= "price",color= "price",
                                 color_continuous_scale= "greens",
                                 title= "BEDROOMS, BEDS, ACCOMMODATES AND BED_TYPE",
                                range_color=(0,df3_top_50_price["price"].max()),
                                width=1200, height= 800,
                                hover_data= ["accommodates","bedrooms","beds","bed_type"])

        st.plotly_chart(fig_top_50_price_2)   

if select == "About":

    st.header("ABOUT THIS PROJECT")

    st.subheader(":orange[1. Data Collection:]")

    st.write('''***Gather data from Airbnb's public API or other available sources.
        Collect information on listings, hosts, reviews, pricing, and location data.***''')
    
    st.subheader(":orange[2. Data Cleaning and Preprocessing:]")

    st.write('''***Clean and preprocess the data to handle missing values, outliers, and ensure data quality.
        Convert data types, handle duplicates, and standardize formats.***''')
    
    st.subheader(":orange[3. Exploratory Data Analysis (EDA):]")

    st.write('''***Conduct exploratory data analysis to understand the distribution and patterns in the data.
        Explore relationships between variables and identify potential insights.***''')
    
    st.subheader(":orange[4. Visualization:]")

    st.write('''***Create visualizations to represent key metrics and trends.
        Use charts, graphs, and maps to convey information effectively.
        Consider using tools like Matplotlib, Seaborn, or Plotly for visualizations.***''')
    
    st.subheader(":orange[5. Geospatial Analysis:]")

    st.write('''***Utilize geospatial analysis to understand the geographical distribution of listings.
        Map out popular areas, analyze neighborhood characteristics, and visualize pricing variations.***''')

