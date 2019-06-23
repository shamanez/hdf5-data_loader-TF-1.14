import tensorflow as tf
from data_set import generator

file_path=''
gen=generator(file_path,"x_sam","y_lab")


dataset = tf.data.Dataset.from_generator(
    gen, 
    tf.uint8)  #check out with the pipe line

dataset = dataset.batch(10)
iterator = dataset.make_one_shot_iterator()

value = dataset.make_one_shot_iterator().get_next()

sess=tf.Session()

data = sess.run(value)
print(data.shape)