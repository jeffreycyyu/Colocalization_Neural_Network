#PREPROCESS
def PREPROCESS_WITH_MASKING(input_array, n_traits):

    #initialize list to store zipped data into
    zipped_padded_test_input = []
    #zip the data to convert 'list of channels which are lists of sequences' into 'lists of sequences which are lists of channels'
    for i in range(int(len(input_array)/n_traits)):
        #zip every set of n_traits together (e.g., if 3 traits/channels, then we want the first 3 sequences representing the three channels for the first gene segment to be zipped, then next 3, ...)
        sub_zipped_padded_test_input = [list(l) for l in zip(input_array[0+i*3], input_array[1+i*3], input_array[2+i*3])]
        #append to master list
        zipped_padded_test_input.append(sub_zipped_padded_test_input)
    # print('preprocessing output') #DELETE_ME
    # print(len(zipped_padded_test_input)) #DELETE_ME
    
    
    
    final_input_list = []
    
    
    for input_sequence_array in zipped_padded_test_input:
        
        #convert to numpy array
        input_sequence_array = np.array(input_sequence_array)
        # print(input_sequence_array.shape)
        # print(len(input_sequence_array))

        # 15% are masked
        #use dstack to have all values in a channel simmultaneously masked or all channels unchanged
        #not all channels will have 15% masked, 15% is for teh maximum length sequences, all others will have less (proportional to sequence length fraction compared to longest sequence length)
        inp_mask = np.random.rand(*input_sequence_array.shape[0:1]) < 0.15
        inp_mask = np.dstack([inp_mask, inp_mask, inp_mask])
        inp_mask = inp_mask[0]
        # print('1') #DELETE ME
        # print(inp_mask) #DELETE ME
        # print(inp_mask.shape)

        # Set targets to -1 by default, it means ignore
        labels = -1 * np.ones(input_sequence_array.shape, dtype=int)
        # print('2') #DELETE ME
        # print(labels) #DELETE ME

        # Set labels for masked tokens
        labels[inp_mask] = input_sequence_array[inp_mask]
        # print('3') #DELETE ME
        # print(labels) #DELETE ME

        # Prepare input
        input_array_masked = np.copy(input_sequence_array)
        # print('4') #DELETE ME
        # print(input_array_masked) #DELETE ME

        # Set input to [MASK] which is the last token for the 90% of tokens
        # This means leaving 10% unchanged
        remain_masked = np.random.rand(*input_sequence_array.shape[0:1]) < 0.90
        remain_masked = np.dstack([remain_masked, remain_masked, remain_masked])
        remain_masked = remain_masked[0]
        inp_mask_2mask = inp_mask & remain_masked
        # print('5') #DELETE ME
        # print(inp_mask_2mask) #DELETE ME

        # print(input_sequence_array.shape) #DELETE ME
        random_array = np.random.uniform(0, 1, input_sequence_array.shape)
        input_array_masked[inp_mask_2mask] = random_array[inp_mask_2mask]  # mask token is the last in the dict
        # print('6') #DELETE ME
        # print(input_array_masked) #DELETE_ME

        #append to output
        final_input_list.append(input_array_masked.tolist())
        
    
    #unzip sequence
    print(len(final_input_list))
    print(len(final_input_list[0]))
    print(len(final_input_list[0][0]))
    #print(final_input_list[0])
    print('unzipping, want 120, then 675 675 675 then something not 675, then 3 3')
    unzipped = [list(zip(*l)) for l in final_input_list]
    unzipped = [list(item) for sublist in unzipped for item in sublist]
    
    #zero pad input list of samples
    # final_input_list = final_input_list.tolist()
    padded_final_input_list = pad_sequences(final_input_list, padding='post', dtype='float32')

    
    return padded_final_input_list



