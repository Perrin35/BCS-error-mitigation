OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4563[3];
rz(5.3321341) q[0];
sx q[0];
rz(3.7254913) q[0];
sx q[0];
rz(13.183594) q[0];
rz(0.33397972) q[1];
sx q[1];
rz(-2.0930891) q[1];
sx q[1];
rz(1.3820678) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-0.56816034) q[2];
sx q[2];
rz(-2.1608278) q[2];
sx q[2];
rz(-1.3481024) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.61722328) q[0];
sx q[0];
rz(-2.557694) q[0];
sx q[0];
rz(0.95105118) q[0];
rz(-1.7934903) q[1];
sx q[1];
rz(2.1608278) q[1];
sx q[1];
rz(-1.3820678) q[2];
sx q[2];
rz(-2.0930891) q[2];
sx q[2];
rz(-0.33397972) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4563[0];
measure q[1] -> c4563[1];
measure q[2] -> c4563[2];
