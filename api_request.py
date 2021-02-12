from qiskit import (QuantumRegister,QuantumCircuit,ClassicalRegister,BasicAer,execute,IBMQ,providers)
from qiskit.visualization import plot_histogram,plot_state_city
import argparse

class notyetdecided:
    def __init__(self,simul=True,token=None):
        self.simul=simul
        self.token=token

    def setup_systems(self):
        if not self.simul:
            assert self.token != None, 'No token passed!'
            IBMQ.save_account(self.token)
            try:
                _providers=IBMQ.load_account()
            except IBMQAccountMultipleCredentialsFound:
                IBMQ.save_account(self.token, overwrite=True)
            finally:        #not sure about this, load_account() twice works but throws a warning in jupyter
                _providers=IBMQ.load_account()
            providers1=[i for i in dir(_providers.backends) if 'ibmq' in i]
#            _f=max(providers1,key=len)
            print('\nAvailable systems for this account: \n')
#            print('\n|{0}||-----||'.format(str(-)*int(_f),))
            for _ in providers1:
                print('|{:<20}|'.format(_),sep='|',end='\n')            #_providers.backends.______.status_msg'.format(_) could've been used for status of systems
            backend0=providers.ibmq.least_busy(_providers.backends(simulator=False)).name()
            print('\nLeast busy system: {}'.format(backend0))
            backend=_providers.get_backend(backend0)
        else:
            backend=BasicAer.get_backend('qasm_simulator')
        print(backend)
y=notyetdecided(False,'024257bbd2ae5845824e3650cd45561fca6cdff0284dc34870113b886144a3ded1c691a8d263d37fb3e5b282a5a5bcdf4573296e632221992acf75cfabb1f65c').setup_systems()
#don't use my key :(
y=notyetdecided(True)
