import os, sys
import tensorflow as tf


import time
start_time = time.clock()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
#image_path = sys.argv[1]
#/tf_files/flowers/data/daisy/99306615_739eb94b9e_m.jpg
image_path = "/tf_files/flowers/data"



def infer( image_path ) :
	itcounter = 0
	# Read in the image_data
	#image_data = tf.gfile.FastGFile(image_path, 'rb').read()

	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line 
	           in tf.gfile.GFile("retrained_labels.txt")]
	# Unpersists graph from file
	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
	with tf.Session() as sess:
		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		for root, dirs, files in os.walk("/tf_files/flowers/data"):
			for filename in files:
				if filename.endswith(".jpg"):
					#print(os.path.join(root, filename))
					image_time = time.clock()
					image_path = os.path.join(root, filename)
					image_data = tf.gfile.FastGFile(image_path, 'rb').read()
					predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
					# Sort to show labels of first prediction in order of confidence
					top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
					for node_id in top_k:
						human_string = label_lines[node_id]
						score = predictions[0][node_id]
						#print('%s (score = %.5f)' % (human_string, score))
						processespersec = 1 / (time.clock() - image_time)
						tputsec = processespersec * 70 
						itcounter = itcounter + 1
						if (itcounter > 100):
							print( "--- %s images per second ---" % (processespersec)) 
							print( "--- %s KB per second ---" % (tputsec)) 
							itcounter = 0
	return


infer( image_path )
print("--- %s seconds ---" % (time.clock() - start_time))
