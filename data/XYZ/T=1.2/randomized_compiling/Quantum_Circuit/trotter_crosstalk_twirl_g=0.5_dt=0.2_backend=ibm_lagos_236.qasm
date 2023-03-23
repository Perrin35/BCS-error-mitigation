OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[3];
rz(-pi) q[0];
sx q[0];
rz(0.42975437) q[0];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
sx q[2];
rz(2.7118383) q[2];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(0.032104437) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(2.7736971) q[1];
sx q[1];
x q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(0.46789556) q[0];
sx q[1];
rz(-2.6736971) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.2707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.4635735) q[1];
sx q[1];
rz(-pi) q[1];
rz(1.4635734) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi) q[0];
rz(3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
sx q[1];
rz(-1.4635735) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.4635734) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.69877088) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-1.4987709) q[1];
sx q[1];
rz(-pi) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-1.9428218) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.1987709) q[1];
rz(-pi) q[2];
sx q[2];
rz(-3.0415927) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi) q[0];
sx q[0];
rz(-0.33377235) q[1];
sx q[1];
rz(-0.33377235) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
x q[0];
x q[1];
rz(0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[1];
sx q[1];
rz(-2.8078203) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(0.33377235) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
sx q[0];
rz(-pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-3.0415927) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.1009358) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.7009358) q[1];
sx q[1];
rz(pi) q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(1.5406569) q[0];
rz(-pi) q[1];
sx q[1];
rz(-1.6009358) q[1];
sx q[2];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi) q[0];
sx q[0];
rz(0.66209605) q[1];
sx q[1];
rz(0.66209609) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
sx q[1];
rz(-0.66209605) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-0.66209609) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.6488339) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(0.69275879) q[1];
sx q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-3.0415927) q[1];
rz(pi/2) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-0.99275875) q[0];
sx q[1];
rz(2.1488339) q[1];
x q[2];
rz(-1.4707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
sx q[0];
rz(-pi) q[0];
rz(3.0808218) q[1];
sx q[1];
rz(-pi) q[1];
rz(-0.060770858) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
sx q[2];
rz(-0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.060770854) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(0.060770858) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
x q[0];
rz(pi) q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi) q[0];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-0.1) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.9941992) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-2.3941991) q[1];
sx q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-0.64739352) q[0];
rz(-pi) q[1];
sx q[1];
rz(2.4941991) q[1];
sx q[2];
rz(2.8415927) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
rz(1.6379629) q[1];
sx q[1];
rz(pi) q[1];
rz(-1.5036297) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(-0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(1.5036298) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.5036297) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.71066151) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.6309311) q[1];
sx q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(1.2106615) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.2106616) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.4707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.5913401) q[1];
sx q[1];
rz(pi) q[1];
rz(-1.5502525) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
rz(3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.5502526) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.5502525) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-0.1) q[0];
sx q[0];
rz(3.0415927) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
