import tensorflow as tf

x = tf.constant(35, name='x')
print(x)
y = tf.Variable(x + 5, name='y')

with tf.Session() as session:
    merged = tf.merge_all_summaries()                       #merges summaries collected in default graph
    writer = tf.train.SummaryWriter("logs/", session.graph) #Writes to buffer files / called async hence in loop
    model = tf.initialize_all_variables()                   # init variables x into y
    session.run(model)                                      #compute
    print(session.run(y))
