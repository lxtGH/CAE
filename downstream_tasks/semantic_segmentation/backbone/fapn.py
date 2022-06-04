class FeatureSelectionModule(nn.Module):
    def __init__(self, in_c, out_c, norm="GM"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_attn = nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, bias=False), # without norm and activation
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
        )
        xavier_init(self.conv_attn)
        for m in self.conv.modeuls():       
            if isintance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        def forward(self, x):
            attn = self.sigmoid(self.conv_attn(F.avg_pool2d(x, x.size()[2:])))
            feat = torch.mul(x, attn)
            x = x + feat
            feat = self.conv(x)
            return feat

class FeatureAlign(nn.Module):
    def __init__(self, in_c, out_c, norm=None):
        super(FeatureAlign, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_c, out_c, norm="")
        self.relu = nn.ReLU(inplace=True)
        self.offset = nn.Conv2d(out_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.deform_conv2d = DeformConv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1, deform_groups=8)
    def forward(self, teat_l, feat_s):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up], dim=1))
        feat_align = self.relu(self.deform_conv2d(feat_up, offset))
        return feat_align + feat_arm
