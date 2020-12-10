import requests
import json

def parseURL(paragraphUrl):
    url = paragraphUrl.split(":50070")
    address = url[0]
    vals = url[1].split("/")
    notebook = vals[3]
    paragraph = vals[5].split("?")[0]
    return [address, notebook, paragraph]

def getData(address, notebook, paragraph):
    response = requests.get(address + ":50070/api/notebook/" + notebook + "/paragraph/" + paragraph)
    return response.text

def getTSV(paragraphUrl):
    # This function gets the same url that you get from clicking on "Link this paragraph"
    [address, notebook, paragraph] = parseURL(paragraphUrl)
    response = getData(address,notebook,paragraph)
    return json.loads(response)["body"]["result"]["msg"]

if __name__ == "__main__":
    url = input()
    dic = getTSV(url)
    print(dic.keys())