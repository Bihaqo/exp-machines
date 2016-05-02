import numpy as np
import timeit
import sys
import resource


class Logging:
    def __init__(self, verbose, metrics, log_w_norm=False):
        """
        Parameters
        ----------
        verbose = 0 -- no logging
        verbose = 1 -- fast logging, do not print the output
        verbose = 2 -- fast logging, print everything
        verbose = 3 -- full logging, print everything
        metrics --  dict of functions like f(linear_output, y_true)-> R . 
                    Example: {'mse':mse.loss, 'logloss':logloss.loss}
                    NOTE: sklearn's metrics use inverted order of arguments
        """
        self.verbose = verbose
        self.log_w_norm = log_w_norm
        self.metrics = metrics

        self.loss_hist = {'train': {}, 'valid':{}}
        for stage in self.loss_hist:
            for loss in metrics:
                self.loss_hist[stage][loss] = []
        if log_w_norm:
            self.w_norm_hist = []

        self.time_hist = []
        self.passes_hist = []
        self.iter_hist = []
        self.prev_timestamp = 0
        self.iter_counter = 0
        self.num_epochs = -1

    def disp(self):
        """Returns true if we are going to print stuff on the screen"""
        if self.verbose >= 2:
            return True
        return False

    def before_first_iter(self, train_x, train_y, w, linear_output_h, num_epochs, num_objects_total):
        """Init logging used for optimization process

        This function get called before the first iteration of optimization.
        """

        if self.iter_counter!=0:
            raise ValueError('Logger.before_first_iter() was already called')

        self.num_objects_total = num_objects_total
        self.num_epochs = num_epochs

        if self.verbose >= 1:
            prev_timestamp = timeit.default_timer()
            linear_output = linear_output_h(w, train_x)
            
            loss_value_pairs = []

            # calculate every metric
            for loss in self.metrics:
                loss_function = self.metrics[loss]
                train_loss_array = loss_function(linear_output, train_y)
                train_loss_value = np.sum(train_loss_array)
                self.loss_hist['train'][loss].append(train_loss_value)
                # TODO: accurate default value (0 may be bad choise for logloss)
                self.loss_hist['valid'][loss].append(0)
                loss_value_pairs.append('{}: {}'.format(loss, train_loss_value))       

            if self.log_w_norm:
                w_norm = w.norm()
                self.w_norm_hist.append(w_norm)
                loss_value_pairs.append('w_norm: {}'.format(w_norm))
            
            # print out status line
            status = 'Init train loss ' + ", ".join(loss_value_pairs)

            self.passes_hist.append(0)
            self.iter_hist.append(0)
            self.time_hist.append(0)
            status += ' stats computed in %f seconds' % (timeit.default_timer() - prev_timestamp)
            if self.disp():
                print status
                sys.stdout.flush()

        self.iter_counter += 1
        self.prev_timestamp = timeit.default_timer()

    def after_each_iter(self, epoch_progress,  train_x, train_y, w, linear_output_h, stage='train'):
        '''
        stage = {'train', 'valid'}
        '''
        if self.iter_counter < 1:
            raise ValueError('logger.after_each_iter: check that before_first_iter was called.')

        # TODO: carefully time estimation
        if self.verbose >= 1:
            linear_output = linear_output_h(w, train_x)

            status = ''
            if stage == 'train':
                elapsed = timeit.default_timer() - self.prev_timestamp
                elapsed_from_start = self.time_hist[-1] + elapsed
                self.time_hist.append(elapsed_from_start)
                self.iter_hist.append(self.iter_counter + 1)
                self.passes_hist.append(epoch_progress + 1)
                status += 'Epoch %d/%d: ' % (epoch_progress + 1, self.num_epochs)
                prev_timestamp = timeit.default_timer()
                status += 'memory usage: %s (kb); ' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            # calculate and print out all metrics
            loss_value_pairs = []
            for loss in self.metrics:
                loss_function = self.metrics[loss]
                loss_array = loss_function(linear_output, train_y)
                loss_value = np.sum(loss_array)
                self.loss_hist[stage][loss].append(loss_value)
                loss_value_pairs.append('{}: {}'.format(loss, loss_value))
            if self.log_w_norm:
                w_norm = w.norm()
                self.w_norm_hist.append(w_norm)
                loss_value_pairs.append('w: {}'.format(w_norm))

            status += stage + ': loss ' + ", ".join(loss_value_pairs)

            if stage == 'train':
                # calculate speed stats
                learning_speed = self.passes_hist[-1] * self.num_objects_total / self.time_hist[-1]
                status += 'processed %f ob/s; stats in %f seconds' % (learning_speed, timeit.default_timer() - prev_timestamp)

            if self.disp():
                print status
                sys.stdout.flush()

        self.prev_timestamp = timeit.default_timer()
        if stage == 'train':
            self.iter_counter += 1

    def get_snapshoot(self):
        snapshoot = {'loss_hist': self.loss_hist,
                     'time_hist': self.time_hist,
                     'iter_hist': self.iter_hist,
                     'passes_hist': self.passes_hist,}
        if self.log_w_norm:
            snapshoot['w_norm_hist'] = self.w_norm_hist

        return snapshoot

    def load(self, logs_dict):
        pass
