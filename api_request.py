from qiskit import (QuantumRegister,QuantumCircuit,ClassicalRegister,BasicAer,execute,IBMQ,providers)
from qiskit.visualization import plot_histogram,plot_state_city
import argparse
from argparse import RawTextHelpFormatter
import sys
from colorama import Fore,Style,Back,init
from configparser import ConfigParser
#from qiskit.backends.jobstatus import JOB_FINAL_STATES
import time
init(autoreset=False)
'''
Available systems for this account:

|ibmq_16_melbourne   |
|ibmq_armonk         |
|ibmq_athens         |
|ibmq_lima           |
|ibmq_qasm_simulator |
|ibmq_quito          |
|ibmq_santiago       |
|ibmqx2              |

Least busy system: ibmq_armonk
ibmq_armonk
'''
config=ConfigParser()
config.read('config.ini')
token=config.get('auth','token')
#token='024257bbd2ae5845824e3650cd45561fca6cdff0284dc34870113b886144a3ded1c691a8d263d37fb3e5b282a5a5bcdf4573296e632221992acf75cfabb1f65c'
#get token or something here, declare a glob var
class notyetdecided:
    def __init__(self,simul=True,listsys=False,leastbusy=False,tok=None):
        if simul:
            listsys,leastbusy=False,False
        self.simul=simul
        self.tok=tok
        self.listsys=listsys
        self.leastbusy=leastbusy

    def setup_systems(self,system=None,needsystem=True):
        if not self.simul:
            assert self.tok != None, 'No token passed!'
            IBMQ.save_account(self.tok)
            try:
                _providers=IBMQ.load_account()
            except:
                print('[{0}!{1}]Error occured during \'IBMQ.load_account()\''.format(Fore.RED,Fore.RESET))
                sys.exit(0)
            if self.listsys:
                providers1=[i for i in dir(_providers.backends) if 'ibmq' in i]
    #            _f=max(provimdders1,key=len)
                print('\nAvailable systems for this account: \n')
    #            print('\n|{0}||-----||'.format(str(-)*int(_f),))
                for _ in providers1:
                    k=_providers.get_backend(_)
                    print('|{}{:<20}{}| {}{:<2}{} |'.format(Fore.MAGENTA,_,Fore.RESET,Fore.WHITE,k.configuration().n_qubits,Fore.RESET),sep='|',end='\n')            #_providers.backends.______.status_msg'.format(_) could've been used for status of systems
                backend0=providers.ibmq.least_busy(_providers.backends(simulator=False)).name()
                print('\nLeast busy system: {0}{1}{2}'.format(Fore.GREEN,backend0,Fore.RESET))
                sys.exit(0)
            try:
                if self.leastbusy:
                    #for _ in _providers.backends(simulator=False).__iter__(): #remove smaller than 4 qubit systems for least_busy()
                    #    if _providers.get_backend(_).configuration().n_qubits>=4:
                    _filteredproviders=[_ for _ in _providers.backends(simulator=False).__iter__() if _providers.get_backend(str(_)).configuration().n_qubits>=4]
                    #backend0=providers.ibmq.least_busy(_providers.backends(simulator=False))
                    backend0=providers.ibmq.least_busy(_filteredproviders).name()
                    print('\nLeast busy system: {0}{1}{2}\n'.format(Fore.GREEN,backend0,Fore.RESET))
                    backend=_providers.get_backend(backend0)
                if needsystem:
                    backend=_providers.get_backend(system)
            except:  #don't know the error :(
                print('[{0}!{1}]Error while selecting backend.'.format(Fore.RED,Fore.Reset))
        else:
            backend=BasicAer.get_backend('qasm_simulator')
        print('Backend selected: {0}{1} ({2}q){3}'.format(Fore.LIGHTBLUE_EX,backend.name(),backend.configuration().n_qubits,Fore.RESET))
        return backend;

#y=notyetdecided(False,'024257bbd2ae5845824e3650cd45561fca6cdff0284dc34870113b886144a3ded1c691a8d263d37fb3e5b282a5a5bcdf4573296e632221992acf75cfabb1f65c').setup_systems()
#don't use my key :(
#y=notyetdecided(True)

def runtime_mode():
    helpf='bruh, give a system name. Use \'-ls\' or \'--list-system\' to list available systems.'
    parser=argparse.ArgumentParser(prog='timemachine',prefix_chars='-',formatter_class=RawTextHelpFormatter,epilog='VOADG')
    yes=parser.add_argument_group('yes')
    yes.add_argument('-yv','--yesv',dest='backend',action='store',help=helpf) #everyone gets one :)
    yes.add_argument('-yo','--yesowen',dest='backend',action='store',help=helpf)
    yes.add_argument('-ya','--yesangy',dest='backend',action='store',help=helpf)
    yes.add_argument('-yd','--yesdavid',dest='backend',action='store',help=helpf)
    yes.add_argument('-yg','--yesgabor',dest='backend',action='store',help=helpf)
    parser.add_argument('-lb','--least-busy',dest='lb',action='store_true',help='Use the least busy system.')
    parser.add_argument('-ls','--list-system',dest='ls',action='store_true',help='List the available systems and also the least busy system.')
    parser.add_argument('-sim','--simulator',dest='siml',action='store_true',help='Use qasm_simulator.')
    parser.add_argument('-t','--token',dest='toke',action='store',help='Token to use and saves to config.')

    args_parsed=parser.parse_args()
    if args_parsed.toke:
        #globals['token']=args_parsed.toke
        config.set('auth','token',args_parsed.toke)
        config.write(sys.stdout)
        print('Token loaded in \'{0}config.ini{1}\''.format(Fore.LIGHTRED_EX,Fore.RESET))
    if not args_parsed.lb and not args_parsed.ls and not args_parsed.backend and not args_parsed.siml:
        print('\n[{0}!{1}]No args parsed.'.format(Fore.RED,Fore.RESET))
        sys.exit(0)
    if args_parsed.lb:
        lb=notyetdecided(simul=False,leastbusy=True,tok=token).setup_systems(needsystem=False)
        #write the functions here
    if args_parsed.ls:
        ls=notyetdecided(simul=False,listsys=True,tok=token).setup_systems(needsystem=False)
    if args_parsed.backend and not args_parsed.siml:
        beekend=notyetdecided(simul=False,tok=token).setup_systems(args_parsed.backend,needsystem=True)
    if args_parsed.siml:
        simm=notyetdecided(simul=True).setup_systems(needsystem=False)

# Q R N G
'''
start_time = time.time()
job_status = job.status()
while job_status not in JOB_FINAL_STATES:
    print(f'Status @ {time.time()-start_time:0.0f} s: {job_status.name},'
          f' est. queue position: {job.queue_position()}')
    time.sleep(10)
    job_status = job.status()'''

#def QRNG():

runtime_mode()