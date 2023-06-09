OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4701[3];
rz(1.0184107) q[0];
sx q[0];
rz(-0.15801604) q[0];
sx q[0];
rz(1.0218793) q[0];
rz(3.1351383) q[1];
sx q[1];
rz(4.0051431) q[1];
sx q[1];
rz(13.219781) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
rz(0.99957543) q[2];
sx q[2];
rz(-1.7571176) q[2];
sx q[2];
rz(-0.97099171) q[2];
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
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.1197134) q[0];
sx q[0];
rz(-0.15801604) q[0];
sx q[0];
rz(-1.0184107) q[0];
rz(0.97099171) q[1];
sx q[1];
rz(-1.7571176) q[1];
sx q[1];
rz(-2.4881823) q[2];
sx q[2];
rz(-0.8635504) q[2];
sx q[2];
rz(0.006454365) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4701[0];
measure q[1] -> c4701[1];
measure q[2] -> c4701[2];
