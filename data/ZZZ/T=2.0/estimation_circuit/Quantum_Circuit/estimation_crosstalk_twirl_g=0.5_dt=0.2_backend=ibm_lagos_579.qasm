OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[3];
rz(3.79073939629478) q[0];
sx q[0];
rz(5.17464657601411) q[0];
sx q[0];
rz(9.66777798267266) q[0];
rz(2.68429637364151) q[1];
sx q[1];
rz(3.63444229114591) q[1];
sx q[1];
rz(13.5815545046416) q[1];
rz(3.36748342167530) q[2];
sx q[2];
rz(5.55465143027241) q[2];
sx q[2];
rz(12.4521622236893) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
x q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi) q[1];
x q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
rz(pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(pi) q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi) q[1];
x q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
x q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
rz(pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
x q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
x q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi) q[0];
x q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
x q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
x q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(pi) q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
x q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
sx q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
x q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi) q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
x q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
rz(pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
rz(pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(pi) q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(pi) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(pi) q[1];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
x q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
x q[1];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi) q[1];
x q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
x q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi) q[0];
x q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
x q[1];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
x q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.243000021903284) q[0];
sx q[0];
rz(1.10853873116548) q[0];
sx q[0];
rz(5.63403856447460) q[0];
rz(-4.15677654387225) q[1];
sx q[1];
rz(2.64874301603368) q[1];
sx q[1];
rz(6.74048158712787) q[1];
rz(-3.02738426291996) q[2];
sx q[2];
rz(0.728533876907175) q[2];
sx q[2];
rz(6.05729453909408) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
