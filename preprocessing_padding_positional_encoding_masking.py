#PREPROCESS
def PREPROCESS(input_array, n_traits):

    #initialize list to store zipped data into
    zipped_padded_test_input = []
    #zip the data to convert 'list of channels which are lists of sequences' into 'lists of sequences which are lists of channels'
    for i in range(int(len(input_array)/n_traits)):
        #zip every set of n_traits together (e.g., if 3 traits/channels, then we want the first 3 sequences representing the three channels for the first gene segment to be zipped, then next 3, ...)
        sub_zipped_padded_test_input = [list(l) for l in zip(input_array[0+i*3], input_array[1+i*3], input_array[2+i*3])]
        #append to master list
        zipped_padded_test_input.append(sub_zipped_padded_test_input)    
    
    #initialize list to store masked data into
    final_input_list = []
    #mask every sequence (a sequence a single LD block of one trait; total number of sequences is number of traits * number of LD blocks)
    for input_sequence_array in zipped_padded_test_input:
        
        #convert to numpy array
        input_sequence_array = np.array(input_sequence_array)

        #15% are masked
        inp_mask = np.random.rand(*input_sequence_array.shape[0:1]) < 0.15
        #use dstack to have all values in a channel simmultaneously masked or all channels unchanged
        inp_mask = np.dstack([inp_mask, inp_mask, inp_mask])
        inp_mask = inp_mask[0]

        #set targets to -1 by default, meaning 'to ignore'
        labels = -1 * np.ones(input_sequence_array.shape, dtype=int)

        #set labels for masked tokens
        labels[inp_mask] = input_sequence_array[inp_mask]

        #prepare input
        input_array_masked = np.copy(input_sequence_array)


        #set input to [MASK] which is the last token for the 90% of tokens (leave 10% unchanged; representing the unmodified 10% of 15% loci that are masked)
        remain_masked = np.random.rand(*input_sequence_array.shape[0:1]) < 0.90
        remain_masked = np.dstack([remain_masked, remain_masked, remain_masked])
        remain_masked = remain_masked[0]
        inp_mask_2mask = inp_mask & remain_masked
        
        #set 10% to a random number between 0 and 1
        random_array = np.random.uniform(0, 1, input_sequence_array.shape)
        input_array_masked[inp_mask_2mask] = random_array[inp_mask_2mask]  # mask token is the last in the dict
        
        #append to output
        final_input_list.append(input_array_masked.tolist())
        
    
    #unzip sequence for zero-padding step
    unzipped = [list(zip(*l)) for l in final_input_list]
    unzipped = [list(item) for sublist in unzipped for item in sublist]
    
    #zero pad input list of samples
    padded_final_input_list = pad_sequences(unzipped, padding='post', dtype='float32')

    #return back into list
    padded_final_input_list = padded_final_input_list.tolist()
    
    #initialize list to store zipped data into
    final_zipped_padded_test_input = []
    #zip the data to convert 'list of channels which are lists of sequences' into 'lists of sequences which are lists of channels' again
    for i in range(int(len(padded_final_input_list)/n_traits)):
        #zip every set of n_traits together (e.g., if 3 traits/channels, then we want the first 3 sequences representing the three channels for the first gene segment to be zipped, then next 3, ...)
        sub_final_zipped_padded_test_input = [list(l) for l in zip(padded_final_input_list[0+i*3], padded_final_input_list[1+i*3], padded_final_input_list[2+i*3])]
        #append to master list
        final_zipped_padded_test_input.append(sub_final_zipped_padded_test_input)
    
    
    #output: masked and zero padded data in a list with the shape [number of LD blocks, maximum number of loci (equal length since zero paddng is preformed), number of traits]
    return final_zipped_padded_test_input
