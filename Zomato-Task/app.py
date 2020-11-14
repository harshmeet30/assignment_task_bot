from flask import Flask, render_template, request

import random
import json
import requests
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from bs4 import BeautifulSoup
import requests


import torch

app = Flask(__name__)

#cities=['Sangli','Abohar ', 'Achalpur ', 'Adilabad ', 'Adityapur ', 'Adoni ', 'Agartala ', 'Agra ', 'Ahmadabad ', 'Ahmadnagar ', 'Aizawl ', 'Ajmer ', 'Akbarpur ', 'Akola ', 'Alandur ', 'Alappuzha ', 'Aligarh ', 'Allahabad ', 'Alwar ', 'Ambala ', 'Ambala Sadar ', 'Ambarnath ', 'Ambattur ', 'Ambikapur ', 'Ambur ', 'Amravati ', 'Amreli ', 'Amritsar ', 'Amroha ', 'Anand ', 'Anantapur ', 'Anantnag ', 'Arrah ', 'Asansol ', 'Ashoknagar Kalyangarh ', 'Aurangabad ', 'Aurangabad ', 'Avadi ', 'Azamgarh ', 'Badlapur ', 'Bagaha ', 'Bagalkot ', 'Bahadurgarh ', 'Baharampur ', 'Bahraich ', 'Baidyabati ', 'Baleshwar Town ', 'Ballia ', 'Bally ', 'Bally City', 'Balurghat ', 'Banda ', 'Bankura ', 'Bansberia ', 'Banswara ', 'Baran ', 'Baranagar ', 'Barasat ', 'Baraut ', 'Barddhaman ', 'Bareilly ', 'Baripada Town ', 'Barnala ', 'Barrackpur ', 'Barshi ', 'Basirhat ', 'Basti ', 'Batala ', 'Bathinda ', 'Beawar ', 'Begusarai ', 'Belgaum ', 'Bellary ', 'Bengaluru', 'Bettiah ', 'Betul ', 'Bhadrak ', 'Bhadravati ', 'Bhadreswar ', 'Bhagalpur ', 'Bhalswa Jahangir Pur ', 'Bharatpur ', 'Bharuch ', 'Bhatpara ', 'Bhavnagar ', 'Bhilai Nagar ', 'Bhilwara ', 'Bhimavaram ', 'Bhind ', 'Bhiwadi ', 'Bhiwandi ', 'Bhiwani ', 'Bhopal ', 'Bhubaneswar Town ', 'Bhuj ', 'Bhusawal ', 'Bid ', 'Bidar ', 'Bidhan Nagar ', 'Biharsharif ', 'Bijapur ', 'Bikaner ', 'Bilaspur ', 'Bokaro Steel City ', 'Bongaon ', 'Botad ', 'Brahmapur Town ', 'Budaun ', 'Bulandshahr ', 'Bundi ', 'Burari ', 'Burhanpur ', 'Buxar ', 'Champdani ', 'Chandannagar ', 'Chandausi ', 'Chandigarh ', 'Chandrapur ', 'Chapra ', 'Chas ', 'Chennai ', 'Chhattarpur ', 'Chhindwara ', 'Chikmagalur ', 'Chilakaluripet ', 'Chitradurga ', 'Chittaurgarh ', 'Chittoor ', 'Churu ', 'Coimbatore ', 'Cuddalore ', 'Cuttack ', 'Dabgram ', 'Dallo Pura ', 'Damoh ', 'Darbhanga ', 'Darjiling ', 'Datia ', 'Davanagere ', 'Deesa ', 'Dehradun ', 'Dehri ', 'Delhi ', 'Delhi Cantonment ', 'Deoghar ', 'Deoli ', 'Deoria ', 'Dewas ', 'Dhanbad ', 'Dharmavaram ', 'Dhaulpur ', 'Dhule ', 'Dibrugarh ', 'Dimapur ', 'Dinapur Nizamat ', 'Dindigul ', 'Dum Dum ', 'Durg ', 'Durgapur ', 'Eluru ', 'English Bazar ', 'Erode ', 'Etah ', 'Etawah ', 'Faizabad ', 'Faridabad ', 'Farrukhabad-cum-Fatehgarh ', 'Fatehpur ', 'Firozabad ', 'Firozpur ', 'Gadag-Betigeri ', 'Gandhidham ', 'Gandhinagar ', 'Ganganagar ', 'Gangapur City ', 'Gangawati ', 'Gaya ', 'Ghazipur ', 'Giridih ', 'Godhra ', 'Gokal Pur ', 'Gonda ', 'Gondal ', 'Gondiya ', 'Gorakhpur ', 'Greater Hyderabad ', 'Greater Mumbai ', 'Greater Noida ', 'Gudivada ', 'Gulbarga ', 'Guna ', 'Guntakal ', 'Guntur ', 'Gurgaon ', 'Guwahati ', 'Gwalior ', 'Habra ', 'Hajipur ', 'Haldia ', 'Haldwani-cum-Kathgodam ', 'Halisahar ', 'Hanumangarh ', 'Haora ', 'Hapur ', 'Hardoi ', 'Hardwar ', 'Hassan ', 'Hastsal ', 'Hathras ', 'Hazaribag ', 'Hindaun ', 'Hindupur ', 'Hinganghat ', 'Hisar ', 'Hoshangabad ', 'Hoshiarpur ', 'Hospet ', 'Hosur ', 'Hubli-Dharwad ', 'Hugli-Chinsurah ', 'Ichalkaranji ', 'Imphal ', 'Indore ', 'Jabalpur ', 'Jagadhri ', 'Jagdalpur ', 'Jaipur ', 'Jalandhar ', 'Jalgaon ', 'Jalna ', 'Jalpaiguri ', 'Jamalpur ', 'Jammu ', 'Jamnagar ', 'Jamshedpur ', 'Jamuria ', 'Jaunpur ', 'Jehanabad ', 'Jetpur Navagadh ', 'Jhansi ', 'Jhunjhunun ', 'Jind ', 'Jodhpur ', 'Junagadh ', 'Kadapa ', 'Kaithal ', 'Kakinada ', 'Kalol ', 'Kalyani ', 'Kamarhati ', 'Kancheepuram ', 'Kanchrapara ', 'Kanpur ', 'Kanpur City', 'Karaikkudi ', 'Karawal Nagar ', 'Karimnagar ', 'Karnal ', 'Kasganj ', 'Kashipur ', 'Katihar ', 'Khammam ', 'Khandwa ', 'Khanna ', 'Kharagpur ', 'Khardaha ', 'Khargone ', 'Khora ', 'Khurja ', 'Kirari Suleman Nagar ', 'Kishanganj ', 'Kishangarh ', 'Kochi ', 'Kolar ', 'Kolhapur ', 'Kolkata ', 'Kollam ', 'Korba ', 'Kota ', 'Kozhikode ', 'Krishnanagar ', 'Kulti ', 'Kumbakonam ', 'Kurichi ', 'Kurnool ', 'Lakhimpur ', 'Lalitpur ', 'Latur ', 'Loni ', 'Lucknow ', 'Ludhiana ', 'Machilipatnam ', 'Madanapalle ', 'Madavaram ', 'Madhyamgram ', 'Madurai ', 'Mahbubnagar ', 'Mahesana ', 'Maheshtala ', 'Mainpuri ', 'Malegaon ', 'Malerkotla ', 'Mandoli ', 'Mandsaur ', 'Mandya ', 'Mangalore ', 'Mango ', 'Mathura ', 'Maunath Bhanjan ', 'Medinipur ', 'Meerut ', 'Mira Bhayander ', 'Miryalaguda ', 'Mirzapur-cum-Vindhyachal ', 'Modinagar ', 'Moga ', 'Moradabad ', 'Morena ', 'Morvi ', 'Motihari ', 'Mughalsarai ', 'Muktsar ', 'Munger ', 'Murwara ', 'Mustafabad ', 'Muzaffarnagar ', 'Muzaffarpur ', 'Mysore ', 'Nabadwip ', 'Nadiad ', 'Nagaon ', 'Nagapattinam ', 'Nagaur ', 'Nagda ', 'Nagercoil ', 'Nagpur ', 'Naihati ', 'Nalgonda ', 'Nanded Waghala ', 'Nandurbar ', 'Nandyal ', 'Nangloi Jat ', 'Narasaraopet ', 'Nashik ', 'Navi Mumbai ', 'Navi Mumbai Panvel Raigarh ', 'Navsari ', 'Neemuch ', 'Nellore ', 'New Delhi ', 'Neyveli ', 'Nizamabad ', 'Noida ', 'North Barrackpur ', 'North Dum Dum ', 'Ongole ', 'Orai ', 'Osmanabad ', 'Ozhukarai ', 'Palakkad ', 'Palanpur ', 'Pali ', 'Pallavaram ', 'Palwal ', 'Panchkula ', 'Panihati ', 'Panipat ', 'Panvel ', 'Parbhani ', 'Patan ', 'Pathankot ', 'Patiala ', 'Patna ', 'Pilibhit ', 'Pimpri Chinchwad ', 'Pithampur ', 'Porbandar ', 'Port Blair ', 'Proddatur ', 'Puducherry ', 'Pudukkottai ', 'Pune ', 'Puri ', 'Purnia ', 'Puruliya ', 'Rae Bareli ', 'Raichur ', 'Raiganj ', 'Raigarh ', 'Raipur ', 'Rajahmundry ', 'Rajapalayam ', 'Rajarhat Gopalpur ', 'Rajkot ', 'Rajnandgaon ', 'Rajpur Sonarpur ', 'Ramagundam ', 'Rampur ', 'Ranchi ', 'Ranibennur ', 'Raniganj ', 'Ratlam ', 'Raurkela Industrial Township ', 'Raurkela Town ', 'Rewa ', 'Rewari ', 'Rishra ', 'Robertson Pet ', 'Rohtak ', 'Roorkee ', 'Rudrapur ', 'S.A.S. Nagar ', 'Sagar ', 'Saharanpur ', 'Saharsa ', 'Salem ', 'Sambalpur ', 'Sambhal ', 'Sangli Miraj Kupwad ', 'Santipur ', 'Sasaram ', 'Satara ', 'Satna ', 'Sawai Madhopur ', 'Secunderabad ', 'Sehore ', 'Seoni ', 'Serampore ', 'Shahjahanpur ', 'Shamli ', 'Shikohabad ', 'Shillong ', 'Shimla ', 'Shimoga ', 'Shivpuri ', 'Sikar ', 'Silchar ', 'Siliguri ', 'Singrauli ', 'Sirsa ', 'Sitapur ', 'Siwan ', 'Solapur ', 'Sonipat ', 'South Dum Dum ', 'Srikakulam ', 'Srinagar ', 'Sujangarh ', 'Sultan Pur Majra ', 'Sultanpur ', 'Surat ', 'Surendranagar Dudhrej ', 'Suryapet ', 'Tadepalligudem ', 'Tadpatri ', 'Tambaram ', 'Tenali ', 'Thane ', 'Thanesar ', 'Thanjavur ', 'Thiruvananthapuram ', 'Thoothukkudi ', 'Thrissur ', 'Tiruchirappalli ', 'Tirunelveli ', 'Tirupati ', 'Tiruppur ', 'Tiruvannamalai ', 'Tiruvottiyur ', 'Titagarh ', 'Tonk ', 'Tumkur ', 'Udaipur ', 'Udgir ', 'Udupi ', 'Ujjain ', 'Ulhasnagar ', 'Uluberia ', 'Unnao ', 'Uttarpara Kotrung ', 'Vadodara ', 'Valsad ', 'Varanasi ', 'Vasai Virar City ', 'Vellore ', 'Veraval ', 'Vidisha ', 'Vijayawada ', 'Visakhapatnam', 'Vizianagaram ', 'Warangal ', 'Wardha ', 'Yamunanagar ', 'Yavatmal ']


#cuisines=['Arabian', 'Finger Food', 'Seafood', 'Juices', 'Steak', 'BBQ', 'Mexican', 'Armenian', 'Kerala', 'Rolls', 'Salad', 'Modern Indian', 'Nepalese', 'Tibetan', 'Singaporean', 'Australian', 'Sushi', 'Tea', 'Odia', 'Bangladeshi', 'Chettinad', 'Mongolian', 'South Indian', 'Fusion', 'Frozen Yogurt', 'Balkans', 'Bakery', 'Contemporary', 'Caribbean', 'Slovak', 'Indian', 'Turkish', 'Sandwich', 'Irish', 'Japanese', 'North Indian', 'Ethiopian', 'Liquor', 'Mughlai', 'Himachali', 'Cuisine Varies', 'Iranian', 'European', 'Moroccan', 'Cantonese', 'Italian', 'Bubble Tea', 'Raw Meats', 'Lebanese', 'Charcoal Chicken', 'Bengali', 'Diner', 'Andhra', 'Brazilian', 'Bihari', 'Russian', 'Breakfast', 'Awadhi', 'Latin American', 'Assamese', 'Mangalorean', 'Wraps', 'Malwani', 'Thai', 'Argentine', 'British', 'African', 'Continental', 'Korean', 'Czech', 'Mediterranean', 'Kebab', 'Bohri', 'Burmese', 'Vietnamese', 'Malaysian', 'Ice Cream', 'Relief fund', 'Mishti', 'Desserts', 'Sri Lankan', 'Roast Chicken', 'Peruvian', 'Rajasthani', 'Egyptian', 'Tex-Mex', 'Maharashtrian', 'North Eastern', 'Coffee and Tea', 'Mithai', 'Pakistani', 'Southern', 'Vegetarian', 'Jamaican', 'Greek', 'Tapas', 'Middle Eastern', 'Goan', 'Tamil', 'Afghan', 'Israeli', 'Burger', 'Spanish', 'Naga', 'Poké', 'Bar Food', 'Sindhi', 'Street Food', 'Portuguese', 'Pizza', 'Biryani', 'Chinese', 'Indonesian', 'Gujarati', 'Konkan', 'International', 'American', 'Hyderabadi', 'Deli', 'French', 'Parsi', 'Others', 'Fast Food', 'Momos', 'Kashmiri', 'Lucknowi', 'German', 'Drinks Only', 'Grill', 'Paan', 'South American', 'Garhwali', 'Hot dogs', 'Panini', 'Beverages', 'Cafe', 'Coffee', 'Grocery', 'Belgian', 'Asian', 'Healthy Food']

headers = {'user-key': 'a60df80ddb6a197f5e37a95238aa3432',
           'Accept': 'application/json'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
bot_name = "BOT"

ist_tags=None




def location(location_name):
    data = {'query': location_name}
    url = 'https://developers.zomato.com/api/v2.1/locations'
    data = requests.post(url, headers=headers, params=data)
    data = json.loads(data.text)

    if(len(data["location_suggestions"])>0):
        entity_type = data["location_suggestions"][0]["entity_type"]
        entity_id = data["location_suggestions"][0]["entity_id"]
        title = data["location_suggestions"][0]["title"]
        city_id=data["location_suggestions"][0]["city_id"]
        country_id=data["location_suggestions"][0]["country_id"]
        country_name=data["location_suggestions"][0]["country_name"]
        details={"restaurants_available":"yes","entity_type":entity_type,"entity_id":entity_id,"title":title,"city_id":city_id,"country_id":country_id,'country_name':country_name}
        return details
    else:
        return {"restaurants_available":"no"}

def location_details(entity_id,entity_type):
    data = {'entity_id': entity_id,"entity_type":entity_type}
    url = 'https://developers.zomato.com/api/v2.1/location_details'
    data = requests.post(url, headers=headers, params=data)
    data = json.loads(data.text)

    nearby_res=data["nearby_res"]
    top_cuisines = data["top_cuisines"]
    best_rated_restaurant=data["best_rated_restaurant"]

    details={"nearby_res":nearby_res,"top_cuisines":top_cuisines,"best_rated_restaurant":best_rated_restaurant}
    
    return details


def cuisine(entity_id,entity_type,cuisine):
    data = {'entity_id': entity_id,"entity_type":entity_type,"q":cuisine}
    url = 'https://developers.zomato.com/api/v2.1/search'
    data = requests.post(url, headers=headers, params=data)
    data = json.loads(data.text)
    
    return data

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
	#global location_name
	userText = request.args.get('msg')
	sentence = tokenize(userText)
	X = bag_of_words(sentence, all_words)
	X = X.reshape(1, X.shape[0])
	X = torch.from_numpy(X).to(device)
	global ist_tags
	output = model(X)
	_, predicted = torch.max(output, dim=1)
	tag = tags[predicted.item()]
	probs = torch.softmax(output, dim=1)
	prob = probs[0][predicted.item()]
	if prob.item() > 0.50:
		for intent in intents['intents']:
			if tag == "Location":
				location_name=userText
				print(location_name)
				d1=location(location_name)
				if d1['country_name']!='India':
					url="https://www.google.com/search?q=my+location"# Make a GET request to fetch the raw HTML content
					html_content = requests.get(url).text# Parse the html content
					soup = BeautifulSoup(html_content, "lxml")#print(soup.prettify())
					ist_tags = soup.find(class_="BNeawe deIvCb AP7Wnd").get_text()#print(ist_tags)
					location_name=ist_tags				
				d1=location(location_name)
				d=location_details(d1['entity_id'],d1['entity_type'])
				l=[]
				for i in range(len(d['best_rated_restaurant'])):
					l.append(d['best_rated_restaurant'][i]['restaurant']['name'])
					ss=", ".join([str(k)+'\n' for k in l])
				return ss
			if tag=='searchCuisine':
				location_name=userText
				print(location_name)
				d1=location(location_name)
				if d1['country_name']!='India':
					url="https://www.google.com/search?q=my+location"# Make a GET request to fetch the raw HTML content
					html_content = requests.get(url).text# Parse the html content
					soup = BeautifulSoup(html_content, "lxml")#print(soup.prettify())
					ist_tags = soup.find(class_="BNeawe deIvCb AP7Wnd").get_text()#print(ist_tags)
					location_name=ist_tags				
				d1=location(location_name)
				d=location_details(d1['entity_id'],d1['entity_type'])
				d1=location(location_name)
				a=cuisine(d1['entity_id'],d1['entity_type'],userText)
				l1=[]
				for i in range (len(a['restaurants'])):
					l1.append(a['restaurants'][i]['restaurant']['name'])
					ss=", ".join([str(k) for k in l1])
				return ss
			if tag =='greeting':
				return(f"{bot_name}: {random.choice(intent['responses'])}")
			if tag =='goodbye':
				return(f"{bot_name}: {random.choice(intent['responses'])}")	
			if tag =='thanks':
				return(f"{bot_name}: {random.choice(intent['responses'])}")
			if tag =='noanswer':
				return(f"{bot_name}: {random.choice(intent['responses'])}")
			if tag =='options':
				return(f"{bot_name}: {random.choice(intent['responses'])}")
			return (f"{bot_name}: I do not understand...")			
	else:
		return(f"{bot_name}: I do not understand...")

if __name__ == "__main__":
    app.run()
