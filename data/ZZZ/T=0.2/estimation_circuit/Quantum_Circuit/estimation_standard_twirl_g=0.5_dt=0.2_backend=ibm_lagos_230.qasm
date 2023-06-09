OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4730[3];
rz(2.9989575) q[0];
sx q[0];
rz(-2.6326257) q[0];
sx q[0];
rz(-0.45605366) q[0];
rz(0.70944866) q[1];
sx q[1];
rz(5.1616321) q[1];
sx q[1];
rz(12.3717) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-0.51493442) q[2];
sx q[2];
rz(-2.3237938) q[2];
sx q[2];
rz(0.54319905) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.45605366) q[0];
sx q[0];
rz(-0.50896694) q[0];
sx q[0];
rz(0.14263511) q[0];
rz(2.5983936) q[1];
sx q[1];
rz(-2.3237938) q[1];
sx q[1];
rz(0.19467034) q[2];
sx q[2];
rz(-1.1215532) q[2];
sx q[2];
rz(-0.70944866) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4730[0];
measure q[1] -> c4730[1];
measure q[2] -> c4730[2];
