OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4667[3];
rz(2.4049987) q[0];
sx q[0];
rz(-1.3741709) q[0];
sx q[0];
rz(-2.4054542) q[0];
rz(-0.9501541) q[1];
sx q[1];
rz(-0.71680161) q[1];
sx q[1];
rz(-3.0870937) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(3.8364031) q[2];
sx q[2];
rz(5.0588342) q[2];
sx q[2];
rz(12.654317) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
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
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.4054542) q[0];
sx q[0];
rz(-1.3741709) q[0];
sx q[0];
rz(-2.4049987) q[0];
rz(0.087946035) q[1];
sx q[1];
rz(-1.9172416) q[1];
sx q[1];
rz(-0.05449899) q[2];
sx q[2];
rz(-0.71680161) q[2];
sx q[2];
rz(0.9501541) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4667[0];
measure q[1] -> c4667[1];
measure q[2] -> c4667[2];