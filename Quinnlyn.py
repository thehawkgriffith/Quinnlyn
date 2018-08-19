#Quinnlyn Prototype v0.1

#Deep NLP Implementation using RNN (LSTM)

#Data Training using Cornell transcripts


#Imorting the libraries

import numpy as np
import tensorflow as tf
import re
import time


###################Preprocessing the Data

lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')


#Mapping each line and its id by a dictionary

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line.update({_line[0]:_line[4]})
        

#List creation of relevant dependant data

conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
    

#Splitting the Initiations and Responses

initiations = []
responses = []
for conversation in conversations_ids:
    for initiation in range(len(conversation) - 1):
        initiations.append(id2line[conversation[initiation]])
        responses.append(id2line[conversation[initiation + 1]])


#Processing the metadata

def process_data(data):

    data = data.lower()
    data = re.sub(r"i'm", "i am", data)
    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"she's", "she is", data)
    data = re.sub(r"you're", "you are", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"what's", "what is", data)
    data = re.sub(r"where's", "where is", data)
    data = re.sub(r"we're", "we are", data)
    data = re.sub(r"they're", "they are", data)
    data = re.sub(r"\'ll", " will", data)
    data = re.sub(r"\'ve", " have", data)
    data = re.sub(r"\'re", " are", data)
    data = re.sub(r"\'s", " is", data)
    data = re.sub(r"\'d", " would", data)
    data = re.sub(r"won't", "will not", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"can't", "can not", data)
    data = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", data)
    return data


#Processing the Initiations

processed_initiations = []
for initiation in initiations:
    processed_initiations.append(process_data(initiation))
    

#Processing the Initiations

processed_responses = []
for response in responses:
    processed_responses.append(process_data(response))


#Creation of a dictionary mapping common occuring elements

word2count = {}
for initiation in processed_initiations:
    for word in initiation.split():
        if word not in word2count:
            word2count.update({word:1})
        else:
            word2count[word] += 1

for response in processed_responses:
    for word in response.split():
        if word not in word2count:
            word2count.update({word:1})
        else:
            word2count[word] += 1


#Two dictionaries each for initiations and responses to map words to a unique integer

threshold = 20
initiationswords2int = {}
occurence = 0
for word, count in word2count.items():
    if count >= threshold:
        initiationswords2int.update({word:occurence})
        occurence += 1
responseswords2int = {}
occurence = 0

for word, count in word2count.items():
    if count >= threshold:
        responseswords2int.update({word:occurence})
        occurence += 1


#Adding last tokens to these two dictionaries

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    initiationswords2int.update({token:(len(initiationswords2int) + 1)})
for token in tokens:
    responseswords2int.update({token:(len(responseswords2int) + 1)})


#Creating inverse mapped dictionary of responseswordsint

responsesint2words = {w_i: w for w, w_i in responseswords2int.items()}


#Adding EOS to the end of every response

for i in range(len(processed_responses)):
    processed_responses[i] += ' <EOS>' 
    

#Transcripting all initiations and responses into integers

#Replacing all words below threshold with <OUT>

initiations2int = []
for initiation in processed_initiations:
    ints = []
    for word in initiation.split():
        if word not in initiationswords2int:
            ints.append(initiationswords2int['<OUT>'])
        else:
            ints.append(initiationswords2int[word])
    initiations2int.append(ints)    

responses2int = []
for response in processed_responses:
    ints = []
    for word in response.split():
        if word not in responseswords2int:
            ints.append(responseswords2int['<OUT>'])
        else:
            ints.append(responseswords2int[word])
    responses2int.append(ints)


#Sorting initiations and responses by the length of initiations

sorted_processed_initiations = []
sorted_processed_responses = []
for length in range(1, 26):
    for i in enumerate(initiations2int):
        if len(i[1]) == length:
            sorted_processed_initiations.append(initiations2int[i[0]])
            sorted_processed_responses.append(responses2int[i[0]])
            

            
#################### The Seq2Seq Model


#Placeholders for the inputs and the targets

def model_inputs():

    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob


#Preprocessing the targets

def preprocess_targets(targets, word2int, batch_size):
    
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


#Encoder RNN Layer
    
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state


#Decoding the training set

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length,
                        decoding_scope, output_function, keep_prob, batch_size):
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                    attention_option = 'bahdanau', 
                                                                                                                                    num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                      attention_keys,
                                                                      attention_values,
                                                                      attention_score_function,
                                                                      attention_construct_function,
                                                                      name = "att_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)    


#Decoding the test/validation set

def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions



#Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, 
                sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions

# Seq2seq model
def seq2seq_model(inputs, targets, keep_prob, sequence_length, responses_num_words, initiations_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, initiationswords2int, batch_size):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              responses_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, initiationswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([initiations_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         initiations_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         initiationswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions





##########TRAINING THE SEQ2SEQ MODEL



# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       sequence_length,
                                                       len(responseswords2int),
                                                       len(initiationswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       initiationswords2int,
                                                       batch_size)

# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches of initiations responses
def split_into_batches(initiations, responses, batch_size):
    for batch_index in range(0, len(initiations) // batch_size):
        start_index = batch_index * batch_size
        initiations_in_batch = initiations[start_index : start_index + batch_size]
        responses_in_batch = responses[start_index : start_index + batch_size]
        padded_initiations_in_batch = np.array(apply_padding(initiations_in_batch, initiationswords2int))
        padded_responses_in_batch = np.array(apply_padding(responses_in_batch, responseswords2int))
        yield padded_initiations_in_batch, padded_responses_in_batch

# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_processed_initiations) * 0.15)
training_initiations = sorted_processed_initiations[training_validation_split:]
training_responses = sorted_processed_responses[training_validation_split:]
validation_initiations = sorted_processed_initiations[:training_validation_split]
validation_responses = sorted_processed_responses[:training_validation_split]

# Training
print("Quinnlyn: Initializing self-training procedure now.")
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_initiations)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "./chatbot_weights.ckpt" 
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_initiations_in_batch, padded_responses_in_batch) in enumerate(split_into_batches(training_initiations, training_responses, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_initiations_in_batch,
                                                                                               targets: padded_responses_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_responses_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_initiations) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_initiations, validation_responses, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_initiations) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('Quinnlyn: I believe I can speak better now.')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Quinnlyn: My apologies, I can not speak any better as of this moment, I will need to train more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("Quinnlyn: My apologies, I can not speak better anymore. This is the best I can do.")
        break
print("Quinnlyn: Self-training completed.")




#EXECUTING THE SEQ2SEQ MODEL
 
 
 
# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)
 
# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = process_data(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Bye' or question == 'Goodbye' or question == 'good bye':
        break
    question = convert_string2int(question, initiationswords2int)
    question = question + [initiationswords2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if responsesint2words[i] == 'i':
            token = ' I'
        elif responsesint2words[i] == '<EOS>':
            token = '.'
        elif responsesint2words[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + responsesint2words[i]
        answer += token
        if token == '.':
            break
    print('Quinnlyn: ' + answer)