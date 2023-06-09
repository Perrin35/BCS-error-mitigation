OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4649[3];
rz(0.87222197) q[0];
sx q[0];
rz(4.7182408) q[0];
sx q[0];
rz(15.701135) q[0];
rz(-2.1380515) q[1];
sx q[1];
rz(-2.1991706) q[1];
sx q[1];
rz(-0.32389855) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
rz(-0.68515732) q[2];
sx q[2];
rz(-0.65396332) q[2];
sx q[2];
rz(-0.93899526) q[2];
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
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
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
rz(3.134764) q[0];
sx q[0];
rz(-1.5766481) q[0];
sx q[0];
rz(2.2693707) q[0];
rz(-0.93899526) q[1];
sx q[1];
rz(2.4876293) q[1];
sx q[1];
rz(2.8176941) q[2];
sx q[2];
rz(-0.94242208) q[2];
sx q[2];
rz(-1.0035411) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4649[0];
measure q[1] -> c4649[1];
measure q[2] -> c4649[2];
