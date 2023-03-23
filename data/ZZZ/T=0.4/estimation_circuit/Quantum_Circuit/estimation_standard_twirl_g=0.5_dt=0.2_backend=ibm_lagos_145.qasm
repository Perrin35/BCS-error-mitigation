OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4945[3];
rz(1.1595261) q[0];
sx q[0];
rz(-2.9268886) q[0];
sx q[0];
rz(0.0052339205) q[0];
rz(1.1637655) q[1];
sx q[1];
rz(-2.1213236) q[1];
sx q[1];
rz(2.2068688) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(0.68261436) q[2];
sx q[2];
rz(-2.4931252) q[2];
sx q[2];
rz(2.2423778) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
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
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
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
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-3.1363587) q[0];
sx q[0];
rz(-0.2147041) q[0];
sx q[0];
rz(1.9820666) q[0];
rz(2.2068688) q[1];
sx q[1];
rz(-1.0202691) q[1];
sx q[1];
rz(1.9778271) q[1];
rz(-2.2423778) q[2];
sx q[2];
rz(-2.4931252) q[2];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4945[0];
measure q[1] -> c4945[1];
measure q[2] -> c4945[2];
