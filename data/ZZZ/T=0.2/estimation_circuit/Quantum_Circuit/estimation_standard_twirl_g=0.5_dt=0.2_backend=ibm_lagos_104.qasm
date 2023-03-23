OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4604[3];
rz(-1.3262773) q[0];
sx q[0];
rz(-2.4361804) q[0];
sx q[0];
rz(0.70391857) q[0];
rz(1.5249042) q[1];
sx q[1];
rz(5.5679713) q[1];
sx q[1];
rz(14.685592) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
rz(3.6263096) q[2];
sx q[2];
rz(4.6721892) q[2];
sx q[2];
rz(10.942378) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
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
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-2.4376741) q[0];
sx q[0];
rz(-0.70541224) q[0];
sx q[0];
rz(-1.8153154) q[0];
rz(-1.5175996) q[1];
sx q[1];
rz(1.6109961) q[1];
sx q[1];
rz(-2.1192209) q[2];
sx q[2];
rz(-0.715214) q[2];
sx q[2];
rz(-1.5249042) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4604[0];
measure q[1] -> c4604[1];
measure q[2] -> c4604[2];
