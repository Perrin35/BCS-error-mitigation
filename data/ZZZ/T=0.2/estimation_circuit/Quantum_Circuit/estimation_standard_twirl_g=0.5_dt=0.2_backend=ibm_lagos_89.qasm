OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4589[3];
rz(-2.9693169) q[0];
sx q[0];
rz(-2.6069508) q[0];
sx q[0];
rz(0.016688987) q[0];
rz(1.3911927) q[1];
sx q[1];
rz(-0.018298115) q[1];
sx q[1];
rz(-1.6776872) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-1.2394496) q[2];
sx q[2];
rz(-0.41417601) q[2];
sx q[2];
rz(0.731191) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-3.1249037) q[0];
sx q[0];
rz(-0.53464181) q[0];
sx q[0];
rz(-0.17227573) q[0];
rz(-0.731191) q[1];
sx q[1];
rz(0.41417601) q[1];
sx q[1];
rz(-1.6776872) q[2];
sx q[2];
rz(-3.1232945) q[2];
sx q[2];
rz(1.7504) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4589[0];
measure q[1] -> c4589[1];
measure q[2] -> c4589[2];
