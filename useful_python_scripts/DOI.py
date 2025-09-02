import requests, json, re, math, textwrap, time

titles=[
"Impact of Robotic Process Automation on Enterprise Resource Planning Systems",
"Enhancing ERP Responsiveness Through Big Data Technologies: An Empirical Investigation",
"The Role of Artificial Intelligence in Enterprise Resource Planning (Erp) Systems",
"AI-powered business process automation in ERP systems: Transforming enterprise operations",
"Integrating Analytics in Enterprise Systems: A Systematic Literature Review of Impacts and Innovations",
"Enterprise Architecture in the Age of Generative AI: Adapting ERP Systems for Next-Generation Automation",
"Evaluating Enterprise Resource Planning (ERP) Implementation for Sustainable Supply Chain Management",
"ERP System Implementation: Planning, Management, and Administrative Issues",
"Machine learning-driven optimization of enterprise resource planning (ERP) systems: a comprehensive review",
"Research on New Trends and Development Prospects of Enterprise Resource Planning (ERP) Systems",
"Implementation of Demand Forecasting Module of ERP System in Mass Customization Industry—Case Studies",
"AI - Driven Supply Chain Optimization: Enhancing Inventory Management, Demand Forecasting, and Logistics within ERP Systems",
"Data Engineering Solutions: The Impact of AI and ML on ERP Systems and Supply Chain Management",
"Integrated Logistics Management Through ERP System: A Case Study in an Emerging Regional Market",
"Artificial Intelligence-Based Methods for Business Processes: A Systematic Literature Review",
"Enhancing Management Control Through ERP Systems: A Comprehensive Literature Review"
]

def get_doi(title):
    url="https://api.crossref.org/works"
    params={"query.title":title, "rows":5}
    try:
        r=requests.get(url, params=params, timeout=10)
        if r.status_code==200:
            items=r.json()['message']['items']
            if items:
                return items[0].get('DOI')
    except Exception as e:
        return None

doimap={}
for t in titles:
    doi=get_doi(t)
    doimap[t]=doi
print(json.dumps(doimap, indent=2))