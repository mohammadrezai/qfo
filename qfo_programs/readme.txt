
Folder qfo_simulation contains the programs for simulating the propagation of a quantum state of light in Fourier optical systems.

-Run 8fhadamard_3qubits.py to simulate the Hadamard gate associated with Fig. 3 of the paper.
-Run 8fcnot.py to simulate the CNOT gate associated with Fig. 4 of the paper.
-8fhadamard_simple.py simulate a one-qubit Hadamard gate that is experimentally less demanding (simple).

*The programs have the option to export the data in openvdb format.
	Note:
		To be able to use the openvdb export option:
		download and install openvdp: https://github.com/AcademySoftwareFoundation/openvdb
		To build OpenVDB with numpy and python support, use:  
                        cmake -D OPENVDB_BUILD_PYTHON_MODULE=ON -D USE_NUMPY=ON ..
		If you don't have openvdb on your pc and you don't need to export data in openvdb format, set variable useopenvdb to False in file qfosys.py.

	Note:
		The openvdb-files hadamard_3qubits.vdb and cnot.vdb are produced by program 8fhadamard_3qubits.py and 8fcnot.py, respectively. 
		Due to the high resolution of the data, it takes several hours to reproduce these files.
	
	Note: 
		The blender files 8fcnoth.blend and 8fhadamard_3qubits.blend use openvdb-files hadamard_3qubits.vdb and cnot.vdb, respectively, to create the paper's Fig. 3 and Fig. 4. 


######################
######################
Folder qfo_gateoptimization contains the programs for optimizing the SLMs' phases of an 8f-system for realizing various quantum gates. 

-opthadamard_3qubits.py gives the optimal phase variables for the Hadamard gate associated with Fig. 3 of the paper.
-optcnot.py  gives the optimal phase variables for the CNOT gate associated with Fig. 4 of the paper.
-opthadamard_simple.py gives the optimal phase variables for a one-qubit Hadamard gate that is experimentally less demanding (simple).