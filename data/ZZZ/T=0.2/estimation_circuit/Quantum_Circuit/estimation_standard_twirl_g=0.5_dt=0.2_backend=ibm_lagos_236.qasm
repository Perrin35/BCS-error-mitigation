OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4736[3];
rz(2.5468792) q[0];
sx q[0];
rz(5.6703735) q[0];
sx q[0];
rz(11.47784) q[0];
rz(-1.0917911) q[1];
sx q[1];
rz(-1.5420869) q[1];
sx q[1];
rz(-1.4693362) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(0.4713635) q[2];
sx q[2];
rz(-2.1954926) q[2];
sx q[2];
rz(2.4201597) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.0530617) q[0];
sx q[0];
rz(-2.5287809) q[0];
sx q[0];
rz(0.5947135) q[0];
rz(-0.72143296) q[1];
sx q[1];
rz(-0.94610007) q[1];
sx q[1];
rz(1.6722564) q[2];
sx q[2];
rz(-1.5995057) q[2];
sx q[2];
rz(-2.0498015) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4736[0];
measure q[1] -> c4736[1];
measure q[2] -> c4736[2];