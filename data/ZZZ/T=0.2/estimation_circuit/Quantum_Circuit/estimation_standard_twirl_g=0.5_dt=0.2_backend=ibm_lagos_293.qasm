OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4793[3];
rz(0.15887343) q[0];
sx q[0];
rz(-2.944351) q[0];
sx q[0];
rz(-1.443393) q[0];
rz(-0.92364402) q[1];
sx q[1];
rz(-0.25751985) q[1];
sx q[1];
rz(2.1370748) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
rz(-2.1223486) q[2];
sx q[2];
rz(-0.66104889) q[2];
sx q[2];
rz(1.6621592) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(1.6981997) q[0];
sx q[0];
rz(-0.1972417) q[0];
sx q[0];
rz(2.9827192) q[0];
rz(-1.6621592) q[1];
sx q[1];
rz(-0.66104889) q[1];
sx q[1];
rz(-1.0045179) q[2];
sx q[2];
rz(-2.8840728) q[2];
sx q[2];
rz(-2.2179486) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4793[0];
measure q[1] -> c4793[1];
measure q[2] -> c4793[2];
