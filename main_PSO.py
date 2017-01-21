#!/usr/bin/python
import sys

print sys.path
from data_handler import data_handler
import optunity.solvers.solver_registry
import json

# import caffe nets
from nets.echo_net_x import EchoXNetModel

if __name__ == "__main__ PSO":
    data_folder = '/home/amir/echoData/AP4/'
    list_train = data_folder + 'list_train.txt'

    list_test = data_folder + 'list_test.txt'

    # dataH_temp = data_handler('single_value', True)
    x, y = data_handler.read_data_from_list_file(list_file=list_train, image_format='.jpg', load_to_memory=True)

    num_folds = 3
    @optunity.cross_validated(x=x, y=y, num_folds=num_folds)
    def echo_netx_error(x_train, y_train, x_test, y_test,
                        #cnn_layers,
                        kernel_num1=None, kernel_num2=None, kernel_num3=None,
                        kernel_size1=None, kernel_size2=None, kernel_size3=None,
                        fc1=None, fc2=None,
                        gen_count=None):

        cnn_layers = 'three'
        echo_netx_error.cross_valid_counter += 1

        hyper_meta_data = {'model_variant': 'echo_netx',
                           'cnn_layers': cnn_layers,
                           'kernel_num1': int(kernel_num1),
                           'kernel_size1': 2*int(kernel_size1) + 1,
                           'fc1': int(fc1),
                           'fc2': int(fc2),
                           'model_counter': str(echo_netx_error.model_counter),
                           'cv_counter': str(echo_netx_error.cross_valid_counter),
                           'generation_counter': gen_count
                           }
        if cnn_layers == 'two' or cnn_layers == "three":
            kernel2_hyperparams = {'kernel_num2': int(kernel_num2), 'kernel_size2': 2 * int(kernel_size2) + 1}
            hyper_meta_data.update(kernel2_hyperparams)
        if cnn_layers == 'three':
            kernel3_hyperparams = {'kernel_num3': int(kernel_num3), 'kernel_size3': 2 * int(kernel_size3) + 1}
            hyper_meta_data.update(kernel3_hyperparams)

        print hyper_meta_data
        model = EchoXNetModel(hyper_meta_data=hyper_meta_data)
        model.set_data(data=[x_train, y_train, x_test, y_test])
        print 'metadata: ', model.create_meta_data
        validation_accuracy = model.train_validate()

        if echo_netx_error.cross_valid_counter >= num_folds:
            echo_netx_error.model_counter += 1
            echo_netx_error.cross_valid_counter = 0

        return validation_accuracy

    echo_netx_error.model_counter = 1
    echo_netx_error.cross_valid_counter = 0

    # search = {
    #     'kernel_num1': [21, 25], 'kernel_size1': [7, 10],
    #     'kernel_num2': [46, 61], 'kernel_size2': [3, 6],
    #     'kernel_num3': [53, 74], 'kernel_size3': [3, 6],
    #     'fc1': [868, 1112],
    #     'fc2': [660, 796]
    #       }
    # optimal_parameters, details, _ = optunity.minimize_structured(echo_netx_error, search_space=search,
    #                                                               num_evals=100, num_particles=5)

    solver = optunity.solvers.ParticleSwarm(num_particles=6, num_generations=12,
                                            kernel_num1=[21, 25],
                                            kernel_num2=[46, 61],
                                            kernel_num3=[53, 74],
                                            kernel_size1=[7, 10],
                                            kernel_size2=[3, 6],
                                            kernel_size3=[3, 6],
                                            fc1=[868, 1112],
                                            fc2=[660, 796]
                                            )
    optimal_parameters, details, _ = solver.minimize(echo_netx_error)

    print str(optimal_parameters)
    with open('optimal_parameters', 'w') as f:
        json.dump(optimal_parameters, f)

    print str(details)
    with open('details', 'w') as f:
        json.dump(details, f)
    print "Optimal parameters: ", optimal_parameters
    print("Cross-validated error rate: %1.3f" % details.optimum)

    df = optunity.call_log2dataframe(details.call_log)
    df.to_csv('details.csv')

    df = df.sort('value', ascending=False)
    df.to_csv('details_sorted.csv')

    print(df)


    # #what I've been doing this
    # optimal_parameters, details, _ = optunity.minimize(echo_netx_error, num_evals=10,
    #                                                    kernel1=[10, 40], kernel2=[20, 70],
    #                                                    fc1=[100, 1000], fc2=[100, 1000],
    #                                                    solver_name='particle swarm')



    # what I was using (unstructured)
    # phi1 = phi2 = 2.05
    # phi = phi1 + phi2
    # max_speed = 2. / (phi - 2. + sqrt(phi * phi - 4 * phi))
    # solver = optunity.solvers.ParticleSwarm(num_particles=6, num_generations=10, phi1=phi, phi2=phi, max_speed=max_speed,
    #                                         kernel1=[10, 100],
    #                                         kernel2=[10, 100],
    #                                         kernel3=[10, 100],
    #                                         kernel_size1=[1, 10],
    #                                         kernel_size2=[1, 10],
    #                                         kernel_size3=[1, 10],
    #                                         fc1=[100, 1000],
    #                                         fc2=[100, 1000])
    # optimal_parameters, details, _ = solver.minimize(echo_netx_error)


    # #another way (same as the first one, with more control)
    # config = optunity.solvers.ParticleSwarm.suggest_from_box(500,
    #                                                          kernel1=[10, 40],
    #                                                          kernel2=[20, 70],
    #                                                          fc1=[100, 1000],
    #                                                          fc2=[100, 1000])
    # solver = optunity.solvers.ParticleSwarm(**config)
    # solver.minimize(echo_netx_error)



