OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4759[3];
rz(-2.9634715) q[0];
sx q[0];
rz(-1.5177) q[0];
sx q[0];
rz(0.40147335) q[0];
rz(1.1793088) q[1];
sx q[1];
rz(-1.2818203) q[1];
sx q[1];
rz(-2.3458133) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(4.3563991) q[2];
sx q[2];
rz(4.1459485) q[2];
sx q[2];
rz(11.635997) q[2];
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
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.40147335) q[0];
sx q[0];
rz(-1.5177) q[0];
sx q[0];
rz(2.9634715) q[0];
rz(-2.211219) q[1];
sx q[1];
rz(2.1372368) q[1];
sx q[1];
rz(-0.79577934) q[2];
sx q[2];
rz(-1.2818203) q[2];
sx q[2];
rz(-1.1793088) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4759[0];
measure q[1] -> c4759[1];
measure q[2] -> c4759[2];
