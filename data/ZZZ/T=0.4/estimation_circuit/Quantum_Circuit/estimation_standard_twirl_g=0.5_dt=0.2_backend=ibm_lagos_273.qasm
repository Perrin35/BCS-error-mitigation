OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c5073[3];
rz(-2.2523242) q[0];
sx q[0];
rz(-2.1656597) q[0];
sx q[0];
rz(3.0457611) q[0];
rz(-2.425627) q[1];
sx q[1];
rz(-1.4625908) q[1];
sx q[1];
rz(2.6244467) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
rz(-1.6122743) q[2];
sx q[2];
rz(-2.2929895) q[2];
sx q[2];
rz(1.1488553) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
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
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
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
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.095831548) q[0];
sx q[0];
rz(-0.97593295) q[0];
sx q[0];
rz(-0.8892685) q[0];
rz(-2.6244467) q[1];
sx q[1];
rz(-1.4625908) q[1];
sx q[1];
rz(2.425627) q[1];
rz(-1.1488553) q[2];
sx q[2];
rz(2.2929895) q[2];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c5073[0];
measure q[1] -> c5073[1];
measure q[2] -> c5073[2];