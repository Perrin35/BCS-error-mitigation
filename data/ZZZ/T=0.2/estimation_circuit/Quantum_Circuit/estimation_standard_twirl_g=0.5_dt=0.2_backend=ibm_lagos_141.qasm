OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4641[3];
rz(1.0995338) q[0];
sx q[0];
rz(-1.8622104) q[0];
sx q[0];
rz(1.5891231) q[0];
rz(-0.70784637) q[1];
sx q[1];
rz(-0.23910147) q[1];
sx q[1];
rz(2.4371489) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
rz(0.36938169) q[2];
sx q[2];
rz(3.5095635) q[2];
sx q[2];
rz(13.949322) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
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
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(1.5524695) q[0];
sx q[0];
rz(-1.8622104) q[0];
sx q[0];
rz(-1.0995338) q[0];
rz(-1.7586411) q[1];
sx q[1];
rz(-0.36797084) q[1];
sx q[1];
rz(2.4371489) q[2];
sx q[2];
rz(-2.9024912) q[2];
sx q[2];
rz(-2.4337463) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4641[0];
measure q[1] -> c4641[1];
measure q[2] -> c4641[2];
