# Six-leg walking circuit (male CNS), built by walk_circuit_build.py

2655 neurons, 82836 chemical connections (>= 5 synapses).
command (descending) -> premotor -> motor (24 leg joints of the flybody fly); leg sensors -> premotor/motor.

Roles: {'premotor': 2205, 'motor': 302, 'command': 81, 'sensory': 67}

Rules: motor neurons of the coxa / femur / tibia / tarsus muscles of all six legs (fly_motor.MN_JOINT);
premotor = VNC intrinsic or ascending neurons with >= 1% of some motor neuron's input;
command = top 20 descending neurons per leg + top 40 overall by reach (direct + one hop),
plus walking types found among DNa02, DNg13, DNb02, DNa01, DNg100, MDN, oDN1, DNg97, DNb05, DNb06; sensors = the 12 leg proprioceptors
(chordotonal, campaniform, hair plate, SNpp/SApp) per leg with the most synapses into the circuit.

neurons.csv : index, root_id, type, role, leg, side, super_class, joint + direction (motor), top_nt, sign, tau_ms,
              total_input_synapses
weights.npz : post, pre, w = sign x synapses / post's total input (graded, as fly_arm), synapses, neuron_ids
