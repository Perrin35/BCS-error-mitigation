OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4713[3];
rz(-2.8718759) q[0];
sx q[0];
rz(-0.79255896) q[0];
sx q[0];
rz(-1.8689121) q[0];
rz(0.70749736) q[1];
sx q[1];
rz(-1.1603713) q[1];
sx q[1];
rz(2.316554) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
rz(2.3232138) q[2];
sx q[2];
rz(-0.24481122) q[2];
sx q[2];
rz(1.870187) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.2726806) q[0];
sx q[0];
rz(-0.79255896) q[0];
sx q[0];
rz(2.8718759) q[0];
rz(1.2714057) q[1];
sx q[1];
rz(-0.24481122) q[1];
sx q[1];
rz(-0.82503863) q[2];
sx q[2];
rz(-1.9812214) q[2];
sx q[2];
rz(2.4340953) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4713[0];
measure q[1] -> c4713[1];
measure q[2] -> c4713[2];
