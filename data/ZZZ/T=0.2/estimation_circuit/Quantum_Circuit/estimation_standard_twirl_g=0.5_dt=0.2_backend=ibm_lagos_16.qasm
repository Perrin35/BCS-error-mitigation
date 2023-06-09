OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4516[3];
rz(-1.3923108) q[0];
sx q[0];
rz(-2.9647093) q[0];
sx q[0];
rz(-2.793762) q[0];
rz(-2.1856862) q[1];
sx q[1];
rz(-1.9560223) q[1];
sx q[1];
rz(-0.3268614) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
rz(-1.8371591) q[2];
sx q[2];
rz(-3.0907501) q[2];
sx q[2];
rz(-0.17297198) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
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
rz(-pi) q[0];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.34783067) q[0];
sx q[0];
rz(-2.9647093) q[0];
sx q[0];
rz(1.3923108) q[0];
rz(0.17297198) q[1];
sx q[1];
rz(3.0907501) q[1];
sx q[1];
rz(-2.8147313) q[2];
sx q[2];
rz(-1.9560223) q[2];
sx q[2];
rz(2.1856862) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4516[0];
measure q[1] -> c4516[1];
measure q[2] -> c4516[2];
