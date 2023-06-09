OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4575[3];
rz(0.20195247) q[0];
sx q[0];
rz(-2.485712) q[0];
sx q[0];
rz(2.0239305) q[0];
rz(-1.2296039) q[1];
sx q[1];
rz(-2.5060703) q[1];
sx q[1];
rz(-2.526762) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
rz(2.7151675) q[2];
sx q[2];
rz(-2.9521763) q[2];
sx q[2];
rz(1.7396219) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
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
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.1176621) q[0];
sx q[0];
rz(-0.65588061) q[0];
sx q[0];
rz(2.9396402) q[0];
rz(-1.4019707) q[1];
sx q[1];
rz(0.18941634) q[1];
sx q[1];
rz(2.526762) q[2];
sx q[2];
rz(-2.5060703) q[2];
sx q[2];
rz(1.2296039) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4575[0];
measure q[1] -> c4575[1];
measure q[2] -> c4575[2];
