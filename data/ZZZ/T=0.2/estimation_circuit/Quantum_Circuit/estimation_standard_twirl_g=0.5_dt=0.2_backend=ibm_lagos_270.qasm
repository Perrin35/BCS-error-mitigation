OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4770[3];
rz(-1.3716711) q[0];
sx q[0];
rz(-0.37175684) q[0];
sx q[0];
rz(0.53003474) q[0];
rz(2.5374473) q[1];
sx q[1];
rz(-0.46110367) q[1];
sx q[1];
rz(-2.8289997) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.4458218) q[2];
sx q[2];
rz(-0.58882131) q[2];
sx q[2];
rz(1.4842865) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
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
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.6115579) q[0];
sx q[0];
rz(-0.37175684) q[0];
sx q[0];
rz(1.3716711) q[0];
rz(1.4842865) q[1];
sx q[1];
rz(2.5527713) q[1];
sx q[1];
rz(-0.31259299) q[2];
sx q[2];
rz(-0.46110367) q[2];
sx q[2];
rz(-2.5374473) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4770[0];
measure q[1] -> c4770[1];
measure q[2] -> c4770[2];
