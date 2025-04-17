import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
from decomposition import tucker_decomposition_conv_layer, create_truncated_svd_sequential, estimate_ranks

def get_layers(model):
    """Recursively get all layers in a PyTorch model."""
    list_layers = []
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            # If it's a sequential container, recursively get its layers
            list_layers.extend(get_layers(layer))
        else:
            # If it's a single layer, add it to the list
            if (isinstance(layer, torch.nn.Conv1d) or isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear)):
                list_layers.append(layer)
    return list_layers

def hardThreshold(A: torch.Tensor, sparsity):
    '''
    Given a Tensor A and the correponding sparsity, returns a copy in the
    format of numpy array with the constraint applied
    NOTE - IT MOVES THE TENSOR TO THE CPU
    '''
    matrix_A = A.data.cpu().detach().numpy().ravel()    
    if len(matrix_A) > 0:
        threshold = np.percentile(np.abs(matrix_A), (1 - sparsity) * 100.0, method='higher')
        matrix_A[np.abs(matrix_A) < threshold] = 0.0
    matrix_A = matrix_A.reshape(A.shape)
    return matrix_A

def compute_sparsity_for_layers(layer_list):
    """Compute sparsity for each layer in a list of layers."""
    sparsity_info = []
    # When using Neural Network
    for layer in layer_list:
        if hasattr(layer, 'weight'):
            weight = layer.weight.data
            total_elements = weight.numel()
            zero_elements = (weight == 0).sum().item()
            sparsity = zero_elements / total_elements
            sparsity_info.append((layer.__class__.__name__, sparsity, total_elements, zero_elements))
    
    # Print the sparsity information for each layer
    for layer, sparsity, total_elements, zero_elements in sparsity_info:
        print(f'Layer: {layer}, Sparsity: {1-sparsity:.4f}, Total Elements: {total_elements}, Zero Elements: {zero_elements}')

    return sparsity_info

######################## Factorization Functions ########################

def find_largest_layer(model, input_tensor, blocked_layers=[], device="cpu"):
    layer_memory = defaultdict(float)
    layer_names = {}

    # Hook to capture output shapes
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            # Feature map memory
            num_elements = output.numel()
            feature_map_memory = num_elements * 4 # Memory in B
            # Parameter memory
            parameter_memory = sum(p.numel() for p in module.parameters() if p.requires_grad) * 4 # Memory in B
            # Total memory for the layer
            #total_memory = feature_map_memory + parameter_memory
            total_memory = parameter_memory
            layer_memory[id(module)] = total_memory
            print(output.shape, feature_map_memory, parameter_memory, total_memory)

    # Register hooks for all layers
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)) and not id(layer) in blocked_layers:
            hook_handle = layer.register_forward_hook(hook_fn)
            hooks.append(hook_handle)
            layer_names[id(layer)] = name

    # Forward pass
    model(input_tensor.to(device))

    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    if not layer_memory:
        return None, None, None
    
    # Find the largest layer
    largest_layer_id = max(layer_memory, key=layer_memory.get)
    largest_layer_name = layer_names[largest_layer_id]
    largest_layer_size = layer_memory[largest_layer_id]
    
    print(f"Largest layer: {largest_layer_name} with memory usage: {largest_layer_size/ (1024**1):.3f} KB")

    return largest_layer_id, largest_layer_name, largest_layer_size

def find_largest_feature_map(model, input_tensor, blocked_layers=[], device="cpu"):    
    # Hook to track feature map sizes
    feature_map_sizes = defaultdict(float)
    layer_names = {}

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            size_in_bytes = output.nelement() * output.element_size()
            feature_map_sizes[id(module)] = size_in_bytes
            print(output.shape, size_in_bytes, output.nelement(), output.element_size())

    # Register hooks
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)) and not id(layer) in blocked_layers:
            hooks.append(layer.register_forward_hook(hook_fn))
            layer_names[id(layer)] = name

    # Forward pass
    model(input_tensor.to(device))

    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    if not feature_map_sizes:
        return None, None, None

    # Find the largest feature map
    largest_layer_id = max(feature_map_sizes, key=feature_map_sizes.get)
    largest_layer_name = layer_names[largest_layer_id]
    largest_layer_size = feature_map_sizes[largest_layer_id]

    print(f"Largest feature map: {largest_layer_name} with size {largest_layer_size/ (1024**1):.3f} KB")


    return largest_layer_id, largest_layer_name, largest_layer_size

def factorize_largest_layers(model, input_tensor, test_loader=None, device="cpu"):
    # ------- default parameters -------
    default_svd_factorization = 0.1
    default_smallest_size_to_factorize = 80000

    # ---- variables for the cycle -----
    original_list = get_layers(model)
    stop_factorization = False
    blocked_layers = []
    cycle = 0
    # ----------------------------------
    
    while stop_factorization != True:
        list_of_layer = get_layers(model)
        largest_layer = None
        print('Iteration:', cycle)
        cycle += 1

        # Get largest layer
        model.to(device)
        largest_layer_id, largest_layer_name, largest_layer_size = find_largest_layer(model, input_tensor, blocked_layers, device=device)

        # If the factorization can be done proceed otherwise add it to the blocked layers
        for layer in original_list:
            if (id(layer) == largest_layer_id):
                largest_layer = layer

        # If no layer was found: stop the factorization (For now, no recursive factorization)
        if not largest_layer:
            print('Layer not found')
            skip=False
            for layer in list_of_layer:
                if (id(layer) == largest_layer_id):
                    blocked_layers.append(largest_layer_id)
                    skip=True
            if skip:
                continue
            stop_factorization = True
            break

        # If largest layer size is small
        if largest_layer_size < default_smallest_size_to_factorize:
            print('Layer already small', largest_layer_size)
            stop_factorization = True
            break

        # Perform the truncation
        truncated_layer = None
        try:
            if isinstance(largest_layer, nn.Conv2d):
                if 0 in estimate_ranks(largest_layer.cpu()):
                    blocked_layers.append(largest_layer_id)
                    continue
                else:
                    model.to(device)
                    truncated_layer = tucker_decomposition_conv_layer(largest_layer.cpu(), device=device)
            elif isinstance(largest_layer, nn.Linear):
                model.to(device)
                truncated_layer = create_truncated_svd_sequential(largest_layer.cpu(), default_svd_factorization, device=device)
            else:
                blocked_layers.append(largest_layer_id)
        except Exception as e:
            blocked_layers.append(largest_layer_id)
            print(e)

        # If the truncation was successfully, change the layer:
        print(truncated_layer)
        if (truncated_layer != None):
            setattr(model, largest_layer_name, truncated_layer)
            blocked_layers.append(largest_layer_id)
        else:
            blocked_layers.append(largest_layer_id)

        # Define other stop factorization condition

    print("-------------- FINISHED FACTORIZATION --------------")
    model.to(device)
    model.eval()
    if (test_loader != None):
        # TEST
        correct = 0
        total = 0
        pred, actual = [], []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pred = pred + list(predicted.detach().cpu().numpy())
                actual = actual + list(labels.detach().cpu().numpy())
        print(f'Test accuracy: {100 * correct /total}')
        print("----------------------------------------------------")

######################## Define Nest Compresion Step Functions ########################

def select_layer_to_compress(layers_sizes, fragile_list, accuracyAware, forced_compression):
    index_of_largest = -1
    if (accuracyAware and not forced_compression):
        index_arg_max = -1
        arg_max = -1
        
        # find largest not fragile layer
        for index, size in enumerate(layers_sizes):
            if (size > arg_max and not fragile_list[index]):
                index_arg_max = index
                arg_max = size

        if (index_arg_max == -1):
            # all layers are fragile
            forced_compression = True
            index_of_largest = np.argmax(layers_sizes)
        else:
            # found the largest non fragile layer
            index_of_largest = index_arg_max
    else:
        index_of_largest = np.argmax(layers_sizes)
    return index_of_largest, forced_compression

def select_sparsity_constraint(model, layers_list, sparsity_list, index_of_largest, initialCompressionStep, compressionStep):    
    # calculate current sizes
    un, comp, layers_sizes = print_size_model(model, layers_list, sparsity_list)
    
    # values defining the compression steps
    initial_step = initialCompressionStep
    
    # get previous size of the layer
    previous_size = layers_sizes[index_of_largest]

    # to make sure the new compression is "enough", based on the compressionStep
    meaningful_compression = False

    # to avoid infinite loops, we put a cap on the number of iterations
    # reduction grows proportionally to compressionStep so unless very small compression step the upper bound is enough to make meaningful compression
    limit = 30
    j = 0
    while not meaningful_compression:
        current_sparsity = sparsity_list[index_of_largest]

        # reduce layers - step
        if (current_sparsity == 1):
            sparsity_list[index_of_largest] = initial_step
        else:
            sparsity_list[index_of_largest] = current_sparsity - (current_sparsity * compressionStep)
        un, comp, layers_sizes = print_size_model(model, layers_list, sparsity_list)

        current_size = layers_sizes[index_of_largest]

        # stops when we reached a compressed size equals to compression step * previous size or we hitted the bound of iterations
        j+=1
        if (current_size <= previous_size * compressionStep or j > limit):
            meaningful_compression = True
    
    return sparsity_list


#######################################################################################

def perform_compression(model, list_of_fc_layers, list_of_fc_sparsity, learning_rate, num_epochs, train_loader,
                        test_loader,model_device,val_loader=None, model_name=None, given_criterion=None,
                        calculate_inputs=None,calculate_outputs=None, history=False, regularizerParam = 0):
    '''
    model has to be sublass of nn.Module
        check the subclass with: issubclass(sub, sup), return true if sub is sublcass of sup
                                 isinstance(sub_instance, sup), return true if is sub_instance is subclass of sup
    list_of_fc_layers: list of fully connected layer OF THE MODEL (should be a pointer to layer of model)
    list_of_fc_sparsity: list of the sparsity for each fully connected layer
    NOTE - Sparsity applied only to weight of FC, not on bias
    NOTE - The list are modified during execution, so are copied with list.copy() to avoid changing the original list
    '''
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    assert len(list_of_fc_layers) == len(list_of_fc_sparsity), "The lists should be of the same length"
    # asset sparsity between 0 and 1
    valid_sparsity = True
    for sparsity in list_of_fc_sparsity:
        if (sparsity > 1) or (sparsity < 0):
            valid_sparsity = False
    assert valid_sparsity, "The sparsity value must be between 0 and 1"
    list_of_fc_layers = list_of_fc_layers.copy()
    list_of_fc_sparsity = list_of_fc_sparsity.copy()
    # The idea is get the model, set all parameter to not require gradient, set fully connected layer to require gradient,
    # perform training
    
    # disabling parameters
    for name, param in model.named_parameters():
        print("Disabling:", name)
        param.requires_grad = False
    
    # activating fully connected layers only if its sparsity is > 0
    # if a layer has sparsity equal to zero we can override with 0
    # if all sparsity is set to 1, compression is not requested
    firstLayer = None
    sparseTraining = False
    for fc_layer, sparsity in zip(list_of_fc_layers, list_of_fc_sparsity):
        if (sparsity == 1):
            # When using Neural Network
            if (sparseTraining):
                print("Activating:", fc_layer)
                fc_layer.weight.requires_grad = True
                if (hasattr(fc_layer, 'bias') and fc_layer.bias is not None):
                    fc_layer.bias.requires_grad = True

        elif (sparsity > 0):
            # When using Neural Network
            print("Activating:", fc_layer)
            fc_layer.weight.requires_grad = True
            if (hasattr(fc_layer, 'bias') and fc_layer.bias is not None):
                fc_layer.bias.requires_grad = True
            if (firstLayer == None):
                firstLayer = id(fc_layer)
        else:
            # When using Neural Network
            fc_layer.weight = torch.nn.Parameter(torch.zeros_like(fc_layer.weight), requires_grad=False)
            if (hasattr(fc_layer, 'bias') and fc_layer.bias is not None):
                fc_layer.bias.requires_grad = True

            # delete from the list (since no need to update them)
            list_of_fc_layers.remove(fc_layer)
            list_of_fc_sparsity.remove(sparsity)
            
        if (sparsity < 1):
            sparseTraining = True
    
    # activating Batch Normalization after the first layer
    activate_after_first = False
    for name, param in model.named_children():
        if (id(param) == firstLayer):
            activate_after_first = True
        if (activate_after_first and isinstance(param, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))):
            print("Activating:", name)
            param.requires_grad = True
    
    acc = 0
    # TEST - compute accuracy
    accuracyHistory = []
    lastCorrect = 0
    totalPredictions = 0
    numberOfUpdates = len(test_loader)
        
    if not (sparseTraining):
        print("No need to perform compression, all layers's sparsity is set to 1")
    else: # PERFORM TRAINING - COMPRESSION
        
        # set up
        criterion = nn.NLLLoss()
        if given_criterion:
            criterion = given_criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        n_total_steps = len(train_loader)
        
        # to save best results
        best_val_epoch, best_val_loss, best_val_acc, best_acc_epoch = 0, 1e6, 0, 0
        
        for epoch in range(num_epochs):
            
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(model_device)
                #inputs = inputs.to(model_device).float()
                labels = labels.to(model_device)
                                
                # Forward pass
                
                # preforward
                if calculate_inputs:
                    inputs = calculate_inputs(inputs)
                
                # forward
                if calculate_outputs:
                    outputs = calculate_outputs(inputs)
                else:
                    outputs, _ = model.forward(inputs)
                
                # Regularization
                regularizer = 0
                if (regularizerParam != 0):
                    for layer in list_of_fc_layers:
                        regularizer += (torch.norm(layer.weight)**2)
                # Loss
                loss = criterion(outputs, labels) + (regularizer * regularizerParam)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # apply hardthreshold - in the list we have only layer with require_grad = True
                for fc_layer, sparsity in zip(list_of_fc_layers, list_of_fc_sparsity):
                    # When using Neural Network
                    layer = fc_layer.weight.data
                    # When using Fast Inference
                    '''
                    layer = fc_layer.data
                    '''
                    new_layer = hardThreshold(layer, sparsity)
                    with torch.no_grad():
                        # When using Neural Network
                        fc_layer.weight.data = torch.FloatTensor(new_layer).to(model_device)
                        # When using Fast Inference
                        '''
                        fc_layer.data = torch.FloatTensor(new_layer).to(model_device)
                        '''
                
                # print Accuracy
                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
            print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
            # Use Validation Set at each epochs to pick the most 
            if (val_loader and model_name):
                model.eval()
                with torch.no_grad():
                    v_loss = 0
                    n_correct = 0
                    n_samples = 0
                    n_iterations = 0
                    for inputs, labels in test_loader:
                        inputs = inputs.to(model_device)
                        #inputs = inputs.to(model_device).float()
                        labels = labels.to(model_device)
                        # Forward pass
                
                        # preforward
                        if calculate_inputs:
                            inputs = calculate_inputs(inputs)
                        outputs = 0 
                        # forward
                        if calculate_outputs:
                            outputs = calculate_outputs(inputs)
                        else:
                            outputs, _ = model.forward(inputs)
                        
                        # for calculating v_loss
                        loss = criterion(outputs, labels)                       
                        v_loss += loss.item()
                        n_iterations += 1
                        
                        # max returns (value, index)
                        _, predicted = torch.max(outputs.data, 1)
                        n_samples += labels.size(0)
                        n_correct += (predicted == labels).sum().item()
                    
                    # Val test completed, now checking the results
                    v_loss = v_loss/(n_iterations)
                    v_loss = round(v_loss, 5)
                    v_acc = round(100*(n_correct / n_samples), 5)
                    
                    if v_acc >= best_val_acc:
                        torch.save(model.state_dict(), model_name+"_acc_NEW.h5")
                        best_acc_epoch = epoch + 1
                        best_val_acc = v_acc
                    if v_loss <= best_val_loss:
                        torch.save(model.state_dict(), model_name+".h5")
                        best_val_epoch = epoch + 1
                        best_val_loss = v_loss
                    #print(f'Epoch[{epoch+1}]: t_loss: {t_loss} t_acc: {t_acc} v_loss: {v_loss} v_acc: {v_acc}')
                    print(f'Epoch[{epoch+1}]: v_loss: {v_loss} v_acc: {v_acc}')
        
        
        # Use Validation Set at each epochs to pick the most 
        if (val_loader and model_name):
            model.load_state_dict(torch.load(model_name+".h5", map_location='cpu'))
            print('Best model saved at epoch: ', best_val_epoch)
            print('Best acc model saved at epoch: ', best_acc_epoch)
        
        # USING TEST SET TO CHECK ACCURACY
        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(model_device)
                #inputs = inputs.to(model_device).float()
                labels = labels.to(model_device)
                   # Forward pass
                
                # preforward
                if calculate_inputs:
                    inputs = calculate_inputs(inputs)
                outputs = 0 
                # forward
                if calculate_outputs:
                    outputs = calculate_outputs(inputs)
                else:
                    outputs, _ = model.forward(inputs)
                # max returns (value, index)
                
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()                
            acc = 100.0 * n_correct / n_samples
            totalPredictions = n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

        
    result = {
        'correctPredictions': lastCorrect,
        'totalPredictions': totalPredictions,
        'accuracyThroughEpochs': accuracyHistory,
        'numberOfUpdate': numberOfUpdates,
    }
    
    return acc

###################################################################

def compress_NN_models(model, target_size, train_loader, test_loader,
                       val_loader=None, num_epochs=10, learning_rate = 0.001, criterion=None,
                       regularizerParam=0.0, compressionStep = 0.1, initialCompressionStep=0.1, fastCompression = False,
                       modelName = "compressed_model", device="cpu", accuracyAware = True,
                       layersFactorization=True, calculateInputs=None
                      ):

    # check model
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    assert target_size > 0, "The target size (kB) should be greater than 0"
    assert regularizerParam >= 0, "Regularizer term should be equal or greater than 0"
    assert compressionStep > 0, "Compression step should be equal or greater than 0"
    
    print("Target size requested:", target_size, "KB")
    
    model = model.to(device)
    
    # get sparsity of model
        # for layers with sparsity less than 1/2, consider as full
    
    if (layersFactorization):
        model.eval()
        # Get the first batch from the train_loader
        single_sample = None
        for batch_samples, batch_labels in train_loader:
            # Extract the first sample from the batch
            single_sample = batch_samples[0:1]
            break
        input_tensor = torch.randn(single_sample.shape).to(device)
        factorize_largest_layers(model, input_tensor, test_loader=test_loader,device=device)
        model = model.to(device)
        
    model.train()
    layers_list = get_layers(model)
    current_sparsity = compute_sparsity_for_layers(layers_list)
    sparsity_list = []
    fragile_list = []
    accuracy_drops = []
    for x, i in enumerate(current_sparsity):
        sp = 1-i[1]
        if sp >= 0.5:
            sp = 1
        sparsity_list.append(sp)
        fragile_list.append(False)
        accuracy_drops.append(0)

    print("Starting Density of model's parameters:", sparsity_list)
    
    # get sizes of model
    un, comp, layers = print_size_model(model, layers_list, sparsity_list,verbose =True)
    original_sizes = layers.copy()
    layers_sizes = layers.copy()
    
    starting_size = un if comp > un else comp
    print("uncomp:", un / 1000, "KB")
    print("comp:", comp / 1000, "KB")
    
    print("Starting size of the model:", starting_size / 1000, "KB")
    
    if (target_size*1000 >= starting_size):
        print("Target size is already met! No compression performed")
        return starting_size
    
    # Compression step, as we get closer to target size we can reduce the size of the step
    # smaller step will require more iteration but will end with closer size to the target
    initialCompressionStep = 0.1
    # step_decay = 0.1
    
    # Training Details
    if (criterion == None):
        criterion = nn.CrossEntropyLoss()
    
    # starting accuracy
    correct = 0
    total = 0
    pred, actual = [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            if (calculateInputs):
                images = calculateInputs(images)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred = pred + list(predicted.detach().cpu().numpy())
            actual = actual + list(labels.detach().cpu().numpy())
    starting_accuracy = 100 * correct /total

    # accuracy aware compression
    previous_accuracy = starting_accuracy
    accuracy_drop_threshold = 5
    forced_compression = False
    
    # End compression
    final_size = comp
    end = False
    
    # start compressing
    i = 0
    train_after_steps = 20
    
    while(not end):
        
        # get index (depth) of largest layer
        index_of_largest, forced_compression = select_layer_to_compress(layers_sizes, fragile_list, accuracyAware, forced_compression)
        
        # compression step
        sparsity_list = select_sparsity_constraint(model, layers_list, sparsity_list, index_of_largest, initialCompressionStep, compressionStep)
        un, comp, layers_sizes = print_size_model(model, layers_list, sparsity_list)
        
        # we compress once we selected the target sparsity values if we want a faster compression time
        # otherwise we can compress at each step, lower epochs are suggested
        if ((fastCompression and comp / 1000 < target_size) or not fastCompression):
            # compress and save result
            MODEL_NAME_COMPRESSED = modelName + "_" + str(round(comp / 1000))
            model.train()
            accuracy = perform_compression(model, layers_list, sparsity_list, learning_rate, num_epochs,
                                           train_loader, test_loader, device,
                                           val_loader=val_loader, model_name=MODEL_NAME_COMPRESSED, given_criterion=criterion,
                                           calculate_inputs=calculateInputs, calculate_outputs=None,
                                           history=False, regularizerParam = 0)
            
            # Check accuracy drop
            accuracy_drop = previous_accuracy - accuracy
            if (accuracy_drop >= accuracy_drop_threshold):
                fragile_list[index_of_largest] = True
                accuracy_drops[index_of_largest] = accuracy_drop
            previous_accuracy = accuracy
            
            # Load best model saved during compression
            model.load_state_dict(torch.load(MODEL_NAME_COMPRESSED+".h5", map_location='cpu'))
            model.to(device)
            model.eval()
            
        if (comp / 1000 <= target_size):
            final_size = comp
            end = True
            break
        
        # continue reducing sparsity
        print(i, "iteration - ", "Size:", comp, sparsity_list)
        i+=1
    
    return final_size

######################## Utility Functions ########################

def print_size_model(model, list_of_fc_layers, list_of_fc_sparsity, verbose=False):
    '''
    model has to be sublass of nn.Module
        check the subclass with: issubclass(sub, sup), return true if sub is sublcass of sup
                                 isinstance(sub_instance, sup), return true if is sub_instance is subclass of sup
    list_of_fc_layers: list of fully connected layer OF THE MODEL (should be a pointer to layer of model)
    list_of_fc_sparsity: list of the sparsity for each fully connected layer
    '''
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    assert len(list_of_fc_layers) == len(list_of_fc_sparsity), "The lists should be of the same length"
    kb = 1000
    verbose and print("-------------------------------------------------------------------------------------------")
    model_size_no_sparsity = 0
    for param in model.parameters():
        model_size_no_sparsity += param.nelement() * param.element_size()
    for buffer in model.buffers():
        model_size_no_sparsity += buffer.nelement() * buffer.element_size()
    
    total_size_no_sparsity = 0
    total_size_with_sparsity = 0
    total_size_with_sparsity_CSC = 0
    
    size_layer_list = []
    num = 0
    for fc_layer, sparsity in zip(list_of_fc_layers, list_of_fc_sparsity):
        num += 1
        # get size
        verbose and print("Layer " + str(num), fc_layer)
        weight = fc_layer.weight.nelement() * fc_layer.weight.element_size()
        bias = 0
        if (hasattr(fc_layer, 'bias') and fc_layer.bias is not None):
            bias = fc_layer.bias.nelement() * fc_layer.bias.element_size()
        
        # save in no sparsity
        total_size_no_sparsity += weight + bias
        
        # set sparsity
        weight = min(1, 2 * sparsity) * weight
        
        # FROM Representation
        if (sparsity <= 0.5):
            if (isinstance(fc_layer, torch.nn.Conv2d)):
                verbose and print("Layer require additional", fc_layer.weight.shape[0], "variables, total size with 4 bytes:", fc_layer.weight.shape[0]*4 / kb)
                total_size_with_sparsity_CSC += (fc_layer.weight.shape[0]*4) # number of filter
            elif (isinstance(fc_layer, torch.nn.Linear)):
                total_size_with_sparsity_CSC += (fc_layer.weight.shape[1] + 1)*4 # number of column
                verbose and print("Layer require additional", fc_layer.weight.shape[1]+1, "variables, total size with 4 bytes:", (fc_layer.weight.shape[1]+1)*4 / kb)
            
        total_size_with_sparsity_CSC += weight + bias
        
        # save in with sparsity
        total_size_with_sparsity += weight + bias
        
        size_layer_list.append(weight + bias)
        
        # print total - print weight - print bias
        verbose and print("Layer "+str(num)+":\t\t", (weight + bias) / kb,
              "KB, \tweight:\t", weight / kb,
              "KB, \tbias:", bias / kb, "KB")
    
    # print total no sparisty
    verbose and print("Size FC Layer (no sparsity):\t", total_size_no_sparsity / kb,"KB")
    
    # print total with sparsity
    verbose and print("Size FC Layer (with sparsity):\t", total_size_with_sparsity / kb,"KB")
    
    # print model total - total no sparsity
    verbose and print("Total Size no sparsity:\t\t", model_size_no_sparsity / kb ,"KB")
    
    # print model total - total no sparisty + total with sparsity
    model_size_with_sparsity = model_size_no_sparsity - total_size_no_sparsity + total_size_with_sparsity
    verbose and print("Total Size with sparsity:\t", model_size_with_sparsity / kb,"KB")
    
    # print model total - total no sparisty + total with sparsity and CSC
    model_size_with_sparsity_CSC = model_size_no_sparsity - total_size_no_sparsity + total_size_with_sparsity_CSC
    verbose and print("Total Size with sparsity and CSC representation:\t", model_size_with_sparsity_CSC / kb,"KB")
    
    verbose and print("-------------------------------------------------------------------------------------------")
    
    return model_size_with_sparsity, model_size_with_sparsity_CSC, size_layer_list

def print_full_model(model):
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    kb = 1000
    model_size = 0
    for name, param in model.named_parameters():
        layer_size = param.nelement() * param.element_size()
        model_size += layer_size
        print(name,"\t", param.nelement(), "\t", param.element_size(),"\t", layer_size / kb, "KB")
        
    for name, buffer in model.named_buffers():
        layer_size = buffer.nelement() * buffer.element_size()
        model_size += layer_size
        print(name,"\t", layer_size / kb, "KB")
    print("Model Size:", model_size / kb, "KB")

def calculate_accuracy(model, train_loader, test_loader, model_device, calculate_inputs=None, calculate_outputs=None):
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"

    acc = 0

    # TEST - compute accuracy
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(model_device)
            labels = labels.to(model_device)
            # preforward
            if calculate_inputs:
                inputs = calculate_inputs(inputs)
                #inputs = inputs.to(model_device).float()
            outputs = 0 
            # forward
            if calculate_outputs:
                outputs = calculate_outputs(inputs)
            else:
                outputs = model.forward(inputs)
                    
            # max returns (value, index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the train images: {acc} %')

        n_correct = 0
        n_samples = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(model_device)
            labels = labels.to(model_device)
            # preforward
            if calculate_inputs:
                inputs = calculate_inputs(inputs)
                #inputs = inputs.to(model_device).float()
            outputs = 0 
            # forward
            if calculate_outputs:
                outputs = calculate_outputs(inputs)
            else:
                outputs = model.forward(inputs)
            
            # max returns (value, index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')

    return acc

def apply_sparsity(model, list_of_fc_layers, list_of_fc_sparsity, model_device):
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    assert len(list_of_fc_layers) == len(list_of_fc_sparsity), "The lists should be of the same length"
    # asset sparsity between 0 and 1
    valid_sparsity = True
    for sparsity in list_of_fc_sparsity:
        if (sparsity > 1) or (sparsity < 0):
            valid_sparsity = False
    assert valid_sparsity, "The sparsity value must be between 0 and 1"
    
    list_of_fc_layers = list_of_fc_layers.copy()
    list_of_fc_sparsity = list_of_fc_sparsity.copy()
    
    # apply hardthreshold - in the list we have only layer with require_grad = True
    for fc_layer, sparsity in zip(list_of_fc_layers, list_of_fc_sparsity):
        layer = fc_layer.weight.data
        new_layer = hardThreshold(layer, sparsity)
        with torch.no_grad():
            fc_layer.weight.data = torch.FloatTensor(new_layer).to(model_device)

    
########################## Under Testing ##########################

def hardThreshold_gpu_1(A: torch.Tensor, sparsity: float) -> torch.Tensor:
    '''
    Given a Tensor A and the correponding sparsity, returns a copy in the
    format of Float Tensor with the constraint applied
    NOTE - SWAP THE TENSOR TO THE GPU
    NOTE - Quantile does not seems to work with larger tensor
    '''
    A = A.to('cuda')
    matrix_A = A.abs().flatten()
    if len(matrix_A) > 0:
        threshold = torch.quantile(matrix_A, (1 - sparsity), interpolation='higher').item()
        A = torch.where(A.abs() < threshold, torch.tensor(0.0, device='cuda'), A)
    return A

def hardThreshold_gpu_2(A: torch.Tensor, sparsity: float, sample_fraction: float = 0.01) -> torch.Tensor:
    '''
    Given a Tensor A and the correponding sparsity, returns a copy in the
    format of Float Tensor with an approximation of the constraint applied
    NOTE - SWAP THE TENSOR TO THE GPU
    NOTE - Use a sampled threshold, faster but less precise - STHOCASTIC
    '''
    A = A.to('cuda')
    
    # Flatten the tensor and get the absolute values
    matrix_A = A.abs().flatten()
    
    # Sample a subset of the data to estimate the threshold
    num_samples = int(len(matrix_A) * sample_fraction)
    if num_samples > 0:
        sample_indices = torch.randint(len(matrix_A), (num_samples,), device='cuda')
        sample_values = matrix_A[sample_indices]
        threshold = torch.quantile(sample_values, 1 - sparsity, interpolation='higher').item()
        
        # Apply the threshold to the original tensor
        A = torch.where(A.abs() < threshold, torch.tensor(0.0, device='cuda'), A)
    
    return A

###################################################################    


'''

-----------=---#@#-----------------------------------------------------------------------------------------
---------@--@--@@--@@--@+----------------------------------------------------------------------------------
-----@*---------------@--@-=-------------------------------------------------------------------------------
-----+----@*#%@@@%%@-@@---@--------------------------------------------------------------------------------
---@*@--%%--@%%=-+%%@-@%@-=**------------------------------------------------------------------------------
-%--@-@%%%@--**--=@--%@%%%--%-------------------------------------------------------------------------%#---
---*-@%%%%%*@@-@--@-@@%%%%%---=@------@----@@--@@----@%-@@-@@----@*-@%=#--@@-+@#--@@--%--@@-@=@@=*---@@----
@@---%%%%%%@=@%@-%@@+%%%%%%%-@%@*-----@----@@--@-@*--@*-@@--@@---@--@=----@@--@@--@@-----@@---@@----@=-@---
-=--@%%%%%%%%%@=#@@%%%%%%%%%--@-*-----@----@@--@--@@-@*-@@---@--@---@=-@--@@@@------@@@=-@@---@@----@@@@@--
@---#@--@@%%%%%*-----=@@@#%%-=@-------@----@@--@---=@@*-@@---@@@=---@=----@@--@-------+@-@@---@@---@----@@-
%----@--%@%%%@--=*+#@@@@@%%%-@---------@@@@@@-=@-----@*-@@----@@----@@@@@-@@---@@-@@@@@--@@---@@--@@----=@@
--@--@@%%@%%%%-*@#=-@@%%%%%--%-----------------------------------------------------------------------------
-@----@%@%%%%@@--=@@%@@%%%-----=---------------------------------------------------------------------------
--%--@--@@@%-%%@-%@@@=%@@---===-------@@@@@@--@@----@@@@@@-@@%@@---#@@@@--@@----@#-@@@@@@--@@++@@----------
---=---@--@=-=%@%+@-@@=----@----------@----@@-@@------@@---@@--@@--+@-----@#@---@=---@@--@@------@@--------
-----=@-@@------------=---------------@-----@-@@------@@---@@@@----+@*%---@--@@-@=---@@--@@------@@--------
------------@-@@---@--@-=-------------@----@@-@@------@@---@@-@@---+@-----@---@@@=---@@--*@------@---------
------------=--@-@-*-----------------*@@@@@---@@------@@---@@---@@-@@@@@@-@-----@=---@@----@@@@@@----------


'''