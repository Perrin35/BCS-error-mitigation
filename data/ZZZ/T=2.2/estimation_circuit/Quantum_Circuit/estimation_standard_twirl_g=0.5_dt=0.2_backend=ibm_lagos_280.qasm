OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c7780[3];
rz(-2.0239765) q[0];
sx q[0];
rz(-2.3523626) q[0];
sx q[0];
rz(-1.9447755) q[0];
rz(1.4526715) q[1];
sx q[1];
rz(-0.31459756) q[1];
sx q[1];
rz(2.0874138) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
rz(4.6115156) q[2];
sx q[2];
rz(5.0046841) q[2];
sx q[2];
rz(10.266365) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
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
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
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
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
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
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
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
rz(-pi) q[2];
x q[2];
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
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
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
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
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
rz(-pi) q[0];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
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
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
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
x q[0];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
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
rz(-pi) q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
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
rz(-pi) q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
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
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
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
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
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
rz(-pi) q[0];
rz(-pi) q[2];
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
rz(-pi) q[2];
x q[2];
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
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
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
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
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
x q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
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
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.9447755) q[0];
sx q[0];
rz(-0.78923001) q[0];
sx q[0];
rz(-1.1176162) q[0];
rz(2.300006) q[1];
sx q[1];
rz(1.2785012) q[1];
sx q[1];
rz(-2.0874138) q[2];
sx q[2];
rz(-0.31459756) q[2];
sx q[2];
rz(-1.4526715) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c7780[0];
measure q[1] -> c7780[1];
measure q[2] -> c7780[2];
