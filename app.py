from flask import Flask, request 
import subprocess
import json


app = Flask(__name__)


''''''''''
Description: This method allows a user to pass in an image 
and get a json back with the instance segmentation of an image 
'''
@app.route('/get_grit_results', methods=['POST'])
def get_oneformer_results():

    # this is the image that the user passed in 
    image = request.files.get("image")

    # here we will save the image so that we can later process it
    image.save('given_image.jpg')

    # this will allow us to run oneformer and then save it to a json which 
    # will have the output that we need 
    subprocess.run("python demo.py --test-task DenseCap --config-file configs/GRiT_B_DenseCap_ObjectDet.yaml  --input images --output /home/davin123/GRiT/visualization/output.jpg --opts MODEL.WEIGHTS models/grit_b_densecap_objectdet.pth", shell=True)

    # here return the stuff from the .json file 
    json_file = open('//home//davin123//GRiT-1//output.json', 'r')
    data = json.load(json_file)
    return data




if __name__ == '__main__':
    # should run on localhost:5000, else you will need to set the port number
    # to some other number 
    app.run(debug=True, port=5001)