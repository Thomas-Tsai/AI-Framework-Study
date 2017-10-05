###############
# TRAIN MODEL #
###############
%matplotlib inline
print '... training'
# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                           # found
improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        iter = (epoch - 1) * n_train_batches + minibatch_index

        if iter % 100 == 0:
            print 'training @ iter = ', iter
        cost_ij = train_model(minibatch_index)

        if (iter + 1) % validation_frequency == 0:

            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                    in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                    improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [
                    test_model(i)
                    for i in xrange(n_test_batches)
                ]
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of '
                        'best model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                        test_score * 100.))
                
    display.clear_output()
    plt.imshow(tile_raster_images(
    X = layer0.W.get_value(borrow=True),
    img_shape=(25,25),
    tile_shape=(10,10),
    tile_spacing=(1,1)), 
    cmap= cm.Greys_r,
    aspect='auto')
    plt.axis('off')
    plt.title('Layer 0 convolutional filters, training cost: ' + str(test_score * 100))
    plt.show()
    plt.imshow(layer2.W.get_value(borrow=True)[:,:].T, 
    cmap= cm.Greys_r)
    plt.axis('off')
    plt.title('Layer 1 fully connected weights, training cost: ' + str(test_score * 100))   
    plt.show()
    plt.imshow(layer3.W.get_value(borrow=True)[:,:].T, 
    cmap= cm.Greys_r)
    plt.axis('off')
    plt.title('Layer 2 fully connected weights, training cost: ' + str(test_score * 100))   
    plt.show()

    if patience <= iter:
        done_looping = True
        break

end_time = timeit.default_timer()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i, '
        'with test performance %f %%' %
        ((1 - best_validation_loss) * 100., best_iter + 1, (1 - test_score) * 100.))
