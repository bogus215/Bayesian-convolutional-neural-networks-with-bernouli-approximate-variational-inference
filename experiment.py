import sys
import gc

#%% MNIST - Bayes- all
# size = [1,4,32]
# for ind, data_size in enumerate(['1562','12500','50000']):
#
#     script_descriptor = open("train.py", encoding ='utf-8')
#     a_script = script_descriptor.read()
#     sys.argv = ["train.py",'--experiment',f'Bayes_data_{data_size}','--data_size',f'{data_size}',
#                 '--image_size','28','--image_channel','1','--dataset','MNIST',
#                 '--epoch','1500','--batch_size',f'{int(512/size[ind])}']
#
#     try:
#         print('start')
#         exec(a_script)
#         gc.collect()
#     except:
#         print('failed')


# size = [1,4,32]
# ind = 0
# data_size = '1562'
# script_descriptor = open("train.py", encoding='utf-8')
# a_script = script_descriptor.read()
# sys.argv = ["train.py", '--experiment', f'Bayes_data_{data_size}', '--data_size', f'{data_size}',
#             '--image_size', '28', '--image_channel', '1', '--dataset', 'MNIST',
#             '--epoch', '1500', '--batch_size', f'{int(512 / size[ind])}']
#
# try:
#     print('start')
#     exec(a_script)
#     gc.collect()
# except:
#     print('failed')


# size = [1,4,32]
# ind = 1
# data_size = '12500'
# script_descriptor = open("train.py", encoding='utf-8')
# a_script = script_descriptor.read()
# sys.argv = ["train.py", '--experiment', f'Bayes_data_{data_size}', '--data_size', f'{data_size}',
#             '--image_size', '28', '--image_channel', '1', '--dataset', 'MNIST',
#             '--epoch', '1500', '--batch_size', f'{int(512 / size[ind])}']
#
# try:
#     print('start')
#     exec(a_script)
#     gc.collect()
# except:
#     print('failed')

#
# size = [1,4,32]
# ind = 1
# data_size = '50000'
# script_descriptor = open("train.py", encoding='utf-8')
# a_script = script_descriptor.read()
# sys.argv = ["train.py", '--experiment', f'Bayes_data_{data_size}', '--data_size', f'{data_size}',
#             '--image_size', '28', '--image_channel', '1', '--dataset', 'MNIST',
#             '--epoch', '1500', '--batch_size', f'{int(512 / size[ind])}']
#
# try:
#     print('start')
#     exec(a_script)
#     gc.collect()
# except:
#     print('failed')



# #%% MNIST - normal - all
# size = [1,4,32]
# for ind, data_size in enumerate(['1562','12500','50000']):
#
#     script_descriptor = open("train.py", encoding ='utf-8')
#     a_script = script_descriptor.read()
#     sys.argv = ["train.py",'--experiment',f'normal_data_{data_size}','--data_size',f'{data_size}',
#                 '--image_size','28','--image_channel','1','--dataset','MNIST',
#                 '--epoch','1500','--batch_size',f'{int(512/size[ind])}']
#
#     try:
#         print('start')
#         exec(a_script)
#         gc.collect()
#     except:
#         print('failed')