OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4560[3];
rz(0.03641686) q[0];
sx q[0];
rz(-2.4269854) q[0];
sx q[0];
rz(-3.0500291) q[0];
rz(-1.7898281) q[1];
sx q[1];
rz(-1.9527938) q[1];
sx q[1];
rz(-1.2066786) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(-2.5047669) q[2];
sx q[2];
rz(-3.0934083) q[2];
sx q[2];
rz(2.7570951) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.091563596) q[0];
sx q[0];
rz(-2.4269854) q[0];
sx q[0];
rz(-0.03641686) q[0];
rz(2.7570951) q[1];
sx q[1];
rz(0.048184395) q[1];
sx q[1];
rz(1.2066786) q[2];
sx q[2];
rz(-1.9527938) q[2];
sx q[2];
rz(1.7898281) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4560[0];
measure q[1] -> c4560[1];
measure q[2] -> c4560[2];
